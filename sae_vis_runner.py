import math
import random
import re
from collections import defaultdict
from typing import Iterable, List, Union

import einops
import numpy as np
import torch
from jaxtyping import Int
from rich import print as rprint
from rich.table import Table
from sae_lens import SAE
from sae_lens.config import DTYPE_MAP as DTYPES
from torch import Tensor
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

from sae_dashboard.components import (
    ActsHistogramData,
    DecoderWeightsDistribution,
    FeatureTablesData,
    LogitsHistogramData,
)
from sae_dashboard.data_parsing_fns import (
    get_features_table_data,
    get_logits_table_data,
)
from sae_dashboard.feature_data import FeatureData
from sae_dashboard.feature_data_generator import FeatureDataGenerator
from sae_dashboard.sae_vis_data import SaeVisConfig, SaeVisData
from sae_dashboard.sequence_data_generator import SequenceDataGenerator
from sae_dashboard.transformer_lens_wrapper import (
    ActivationConfig,
    TransformerLensWrapper,
)
from sae_dashboard.utils_fns import FeatureStatistics


class FeatureDataGeneratorFactory:
    @staticmethod
    def create(
        cfg: SaeVisConfig,
        model: HookedTransformer,
        encoder: SAE,
        tokens: Int[Tensor, "batch seq"],
    ) -> FeatureDataGenerator:
        """Builds a FeatureDataGenerator using the provided config and model."""
        activation_config = ActivationConfig(
            primary_hook_point=cfg.hook_point,
            auxiliary_hook_points=(
                [
                    re.sub(r"hook_z", "hook_v", cfg.hook_point),
                    re.sub(r"hook_z", "hook_pattern", cfg.hook_point),
                ]
                if cfg.use_dfa
                else []
            ),
        )
        wrapped_model = TransformerLensWrapper(model, activation_config)
        return FeatureDataGenerator(
            cfg=cfg, model=wrapped_model, encoder=encoder, tokens=tokens
        )


class SaeVisRunner:
    def __init__(self, cfg: SaeVisConfig) -> None:
        self.cfg = cfg
        self.device = self.cfg.device
        self.dtype = DTYPES[self.cfg.dtype]
        if self.cfg.cache_dir is not None:
            self.cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    @torch.inference_mode()
    def run(
        self, encoder: SAE, model: HookedTransformer, tokens: Int[Tensor, "batch seq"], edited_mask: Int[Tensor, "batch seq"]
    ) -> SaeVisData:
        # Apply random seed
        self.set_seeds()

        # add extra method to SAE which is not yet provided by SAE Lens.
        # encoder = self.mock_feature_acts_subset_for_now(encoder)
        encoder.fold_W_dec_norm()

        # turn off reshaping mode since that's not useful if we're caching activations on disk
        if encoder.hook_z_reshaping_mode:
            encoder.turn_off_forward_pass_hook_z_reshaping()

        # set precision on encoders and model
        # encoder = encoder.to(DTYPES[self.cfg.dtype])
        # # model = cast(HookedTransformer, model.to(DTYPES[self.cfg.dtype]))

        # model.to(self.cfg.device)
        # encoder = encoder.to(self.cfg.device)
        time_logs = defaultdict(float)

        features_list = self.handle_features(self.cfg.features, encoder)
        feature_batches = self.get_feature_batches(features_list)
        progress = self.get_progress_bar(tokens, feature_batches, features_list)

        feature_data_generator = FeatureDataGeneratorFactory.create(
            self.cfg, model, encoder, tokens
        )

        sequence_data_generator = SequenceDataGenerator(
            cfg=self.cfg,
            tokens=tokens,
            W_U=model.W_U,
        )

        all_consolidated_dfa_results = {
            feature_idx: {} for feature_idx in self.cfg.features
        }


        
        # Create objects to store all the data we'll get from `_get_feature_data`
        sae_vis_data = SaeVisData(cfg=self.cfg)
        sae_vis_data_masked = SaeVisData(cfg=self.cfg)


        # For each batch of features: get new data and update global data storage objects
        # TODO: We should write out json files with the results as this runs rather than storing everything in memory
        for features in feature_batches:
            # model and sae activations calculations.

            (
                all_feat_acts,
                _,  # all resid post. no longer used.
                feature_resid_dir,
                feature_out_dir,
                corrcoef_neurons,
                corrcoef_encoder,
                batch_dfa_results,
            ) = feature_data_generator.get_feature_data(features, progress)
            # can just apply mask to every feature? at this point
            # and abstract the below functions -> Yes

            # APPLY MASKED
            all_feat_acts_masked = all_feat_acts * edited_mask.unsqueeze(-1)

            # sort indices by sums, large to small
            sums = all_feat_acts_masked.sum(dim=(0, 1))
            sorted_indices = torch.argsort(sums, descending=True)

            all_feat_acts_masked = all_feat_acts_masked[:, :, sorted_indices]
            all_feat_acts = all_feat_acts[:, :, sorted_indices]

            # filter out matrices with a sum of 0
            # todo, if work, can combine steps
            non_zero_indices = sums[sorted_indices] != 0  # Boolean mask for non-zero sums
            all_feat_acts_masked = all_feat_acts_masked[:, :, non_zero_indices]
            all_feat_acts = all_feat_acts[:, :, non_zero_indices]
            features = [feat for feat, is_non_zero in zip(features, non_zero_indices) if is_non_zero]
            #print(f"[{feature_batch_idx}] feature_length/sum(acts masked)/sum(acts)={len(features)}/{torch.sum(all_feat_acts_masked).item():.2f}/{torch.sum(all_feat_acts).item():.2f}")
            ## end of applying mask
            if torch.sum(all_feat_acts_masked).item() < 0.01: 
                #print(f"[{feature_batch_idx}] Skipped")
                #features = []
                continue

            def populate(sae_vis_data_object, all_feat_acts_data, corrcoef_flag):
            
                # Get the logits of all features (i.e. the directions this feature writes to the logit output)
                logits = einops.einsum(
                    feature_resid_dir.to(device=model.W_U.device, dtype=model.W_U.dtype),
                    model.W_U,
                    "feats d_model, d_model d_vocab -> feats d_vocab",
                ).to(self.device)

                # ! Get stats (including quantiles, which will be useful for the prompt-centric visualisation)
                feature_stats = FeatureStatistics.create(
                    data=einops.rearrange(all_feat_acts_data, "b s feats -> feats (b s)"),
                    batch_size=self.cfg.quantile_feature_batch_size,
                )

                # ! Data setup code (defining the main objects we'll eventually return)
                feature_data_dict: dict[int, FeatureData] = {
                    feat: FeatureData() for feat in features
                }

                # We're using `cfg.feature_centric_layout` to figure out what data we'll need to calculate during this function
                layout = self.cfg.feature_centric_layout

                feature_tables_data = get_features_table_data(
                    feature_out_dir=feature_out_dir,
                    corrcoef_neurons=corrcoef_neurons if corrcoef_flag else None,
                    corrcoef_encoder=corrcoef_encoder if corrcoef_flag else None,
                    n_rows=layout.feature_tables_cfg.n_rows,  # type: ignore
                )

                # Add all this data to the list of FeatureTablesData objects
                if batch_dfa_results:
                    # Accumulate DFA results across feature batches
                    for feature_idx, feature_data in batch_dfa_results.items():
                        all_consolidated_dfa_results[feature_idx].update(feature_data)

                for i, (feat, logit_vector) in enumerate(zip(features, logits)):
                    # __ for each feature __

                    feature_data_dict[feat].feature_tables_data = FeatureTablesData(
                        **{k: v[i] for k, v in feature_tables_data.items()}  # type: ignore
                    )

                    # Get logits histogram data (no title)
                    feature_data_dict[feat].logits_histogram_data = (
                        LogitsHistogramData.from_data(
                            data=logit_vector.to(
                                torch.float32
                            ),  # need this otherwise fails on MPS
                            n_bins=layout.logits_hist_cfg.n_bins,  # type: ignore
                            tickmode="5 ticks",
                            title=None,
                        )
                    )

                    # Get data for feature activations histogram (including the title!)
                    feat_acts = all_feat_acts_data[..., i]

                    # Create a mask for tokens to ignore based on both ID and position
                    ignore_tokens_mask = torch.ones_like(tokens, dtype=torch.bool)
                    if self.cfg.ignore_tokens:
                        ignore_tokens_mask &= ~torch.isin(
                            tokens,
                            torch.tensor(
                                list(self.cfg.ignore_tokens),
                                dtype=tokens.dtype,
                                device=tokens.device,
                            ),
                        )
                    if self.cfg.ignore_positions:
                        ignore_positions_mask = torch.ones_like(tokens, dtype=torch.bool)
                        ignore_positions_mask[:, self.cfg.ignore_positions] = False
                        ignore_tokens_mask &= ignore_positions_mask

                    # Move the mask to the same device as feat_acts
                    ignore_tokens_mask = ignore_tokens_mask.to(feat_acts.device)

                    # set any masked positions to 0
                    masked_feat_acts = feat_acts * ignore_tokens_mask

                    # Apply the mask to feat_acts
                    nonzero_feat_acts = masked_feat_acts[masked_feat_acts > 0]
                    frac_nonzero = nonzero_feat_acts.numel() / masked_feat_acts.numel()

                    # filter out fraction
                    #print(f"frac_nonzero={frac_nonzero}, density={frac_nonzero*100}%")
                    if frac_nonzero < 1e-3: # less than 0.1%
                        #print("Too small frac. Skipping...")
                        del feature_data_dict[feat]
                        continue
                    feature_data_dict[feat].frac_nonzero = frac_nonzero
                    ##### moving density calculation upfront.
                    ##### determine whether to continue computation by looking at the density *first*

                    feature_data_dict[feat].acts_histogram_data = (
                        ActsHistogramData.from_data(
                            data=nonzero_feat_acts.to(
                                torch.float32
                            ),  # need this otherwise fails on MPS
                            n_bins=layout.act_hist_cfg.n_bins,  # type: ignore
                            tickmode="5 ticks",
                            title=f"ACTIVATIONS<br>DENSITY = {frac_nonzero:.3%}",
                        )
                    )

                    # Create a MiddlePlotsData object from this, and add it to the dict
                    feature_data_dict[feat].logits_table_data = get_logits_table_data(
                        logit_vector=logit_vector,
                        n_rows=layout.logits_table_cfg.n_rows,  # type: ignore
                    )

                    # ! Calculate all data for the right-hand visualisations, i.e. the sequences

                    # Add this feature's sequence data to the list
                    feature_data_dict[feat].sequence_data = (
                        sequence_data_generator.get_sequences_data(
                            feat_acts=masked_feat_acts,
                            feat_logits=logits[i],
                            resid_post=torch.tensor([]),  # no longer used
                            feature_resid_dir=feature_resid_dir[i],
                        )
                    )
                    if self.cfg.use_dfa:
                        feature_data_dict[feat].dfa_data = all_consolidated_dfa_results.get(
                            feat, None
                        )
                        feature_data_dict[feat].decoder_weights_data = (
                            get_decoder_weights_distribution(encoder, model, feat)[0]
                        )

                    # Update the 2nd progress bar (fwd passes & getting sequence data dominates the runtime of these computations)
                    if progress is not None:
                        progress[1].update(1)

                # ! Return the output, as a dict of FeatureData items
                new_feature_data = SaeVisData(
                    cfg=self.cfg,
                    feature_data_dict=feature_data_dict,
                    feature_stats=feature_stats,
                )

                sae_vis_data_object.update(new_feature_data)

            populate(sae_vis_data_object=sae_vis_data, all_feat_acts_data=all_feat_acts, corrcoef_flag=True)
            populate(sae_vis_data_object=sae_vis_data_masked, all_feat_acts_data=all_feat_acts_masked, corrcoef_flag=True)

        # Now exited, make sure the progress bar is at 100%
        if progress is not None:
            for pbar in progress:
                pbar.n = pbar.total

        # If verbose, then print the output
        if self.cfg.verbose:
            total_time = sum(time_logs.values())
            table = Table("Task", "Time", "Pct %")
            for task, duration in time_logs.items():
                table.add_row(task, f"{duration:.2f}s", f"{duration/total_time:.1%}")
            rprint(table)

        sae_vis_data.cfg = self.cfg
        sae_vis_data.model = model
        sae_vis_data.encoder = encoder

        sae_vis_data_masked.cfg = self.cfg
        sae_vis_data_masked.model = model
        sae_vis_data_masked.encoder = encoder
        return sae_vis_data, sae_vis_data_masked

    def set_seeds(self) -> None:
        if self.cfg.seed is not None:
            random.seed(self.cfg.seed)
            torch.manual_seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)
        return None

    def handle_features(
        self, features: Iterable[int] | None, encoder_wrapper: SAE
    ) -> list[int]:
        if features is None:
            return list(range(encoder_wrapper.cfg.d_sae))
        else:
            return list(features)

    def get_feature_batches(self, features_list: list[int]) -> list[list[int]]:
        # Break up the features into batches
        feature_batches = [
            x.tolist()
            for x in torch.tensor(features_list).split(self.cfg.minibatch_size_features)
        ]
        return feature_batches

    def get_progress_bar(
        self,
        tokens: Int[Tensor, "batch seq"],
        feature_batches: list[list[int]],
        features_list: list[int],
    ):
        # Calculate how many minibatches of tokens there will be (for the progress bar)
        n_token_batches = (
            1
            if (self.cfg.minibatch_size_tokens is None)
            else math.ceil(len(tokens) / self.cfg.minibatch_size_tokens)
        )

        # Get the denominator for each of the 2 progress bars
        totals = (n_token_batches * len(feature_batches), len(features_list))

        # Optionally add two progress bars (one for the forward passes, one for getting the sequence data)
        if self.cfg.verbose:
            progress = [
                tqdm(total=totals[0], desc="Forward passes to cache data for vis"),
                tqdm(total=totals[1], desc="Extracting vis data from cached data"),
            ]
        else:
            progress = None

        return progress


def get_decoder_weights_distribution(
    encoder: SAE,
    model: HookedTransformer,
    feature_idx: Union[int, List[int]],
) -> List[DecoderWeightsDistribution]:
    if not isinstance(feature_idx, list):
        feature_idx = [feature_idx]

    distribs = []
    for feature in feature_idx:
        att_blocks = einops.rearrange(
            encoder.W_dec[feature, :],
            "(n_head d_head) -> n_head d_head",
            n_head=model.cfg.n_heads,
        ).to("cpu")
        decoder_weights_distribution = (
            att_blocks.norm(dim=1) / att_blocks.norm(dim=1).sum()
        )
        distribs.append(
            DecoderWeightsDistribution(
                model.cfg.n_heads, [float(x) for x in decoder_weights_distribution]
            )
        )

    return distribs
