import logging

import pytorch_lightning as pl
import torch
import torchvision.transforms as T


class CNOS(pl.LightningModule):
    """
    Container for the two CNOS stages:
      1. `segmentor_model`  (SAM2)   : RGB image -> class-agnostic mask proposals
      2. `descriptor_model` (DINOv3) : proposals -> CLS-token descriptors
    plus the template matching used to assign each proposal to a reference object.
    """

    def __init__(
        self,
        segmentor_model,
        descriptor_model,
        matching_config,
        post_processing_config,
        **kwargs,
    ):
        super().__init__()
        self.segmentor_model = segmentor_model
        self.descriptor_model = descriptor_model
        self.matching_config = matching_config
        self.post_processing_config = post_processing_config
        self.ref_data = {"descriptors": None}

        self.inv_rgb_transform = T.Compose(
            [
                T.Normalize(
                    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
                ),
            ]
        )
        logging.info("Init CNOS done!")

    def move_to_device(self, device=None):
        """Move both backbones to `device` (defaults to this module's device)."""
        device = torch.device(device) if device is not None else self.device
        self.descriptor_model.model = self.descriptor_model.model.to(device)
        self.descriptor_model.model.device = device
        self.segmentor_model.predictor.model = self.segmentor_model.predictor.model.to(
            device
        )
        logging.info(f"Moving models to {device} done!")

    def set_reference_descriptors(self, descriptors):
        """
        descriptors: N_objects x N_templates x D tensor of template descriptors.
        """
        self.ref_data["descriptors"] = descriptors

    def find_matched_proposals(self, proposal_descriptors):
        """
        Match proposal descriptors against the reference descriptors.

        Returns (idx_selected_proposals, pred_idx_objects, pred_scores,
        pred_score_distribution) following the original CNOS matching rule.
        """
        assert (
            self.ref_data["descriptors"] is not None
        ), "call set_reference_descriptors() first"
        scores = self.matching_config.metric(
            proposal_descriptors, self.ref_data["descriptors"]
        )  # N_proposals x N_objects x N_templates
        if self.matching_config.aggregation_function == "mean":
            score_per_proposal_and_object = (
                torch.sum(scores, dim=-1) / scores.shape[-1]
            )  # N_proposals x N_objects
        elif self.matching_config.aggregation_function == "median":
            score_per_proposal_and_object = torch.median(scores, dim=-1)[0]
        elif self.matching_config.aggregation_function == "max":
            score_per_proposal_and_object = torch.max(scores, dim=-1)[0]
        elif self.matching_config.aggregation_function == "avg_5":
            k = min(5, scores.shape[-1])
            score_per_proposal_and_object = torch.topk(scores, k=k, dim=-1)[0]
            score_per_proposal_and_object = torch.mean(
                score_per_proposal_and_object, dim=-1
            )
        else:
            raise NotImplementedError

        # assign each proposal to the object with highest score
        score_per_proposal, assigned_idx_object = torch.max(
            score_per_proposal_and_object, dim=-1
        )  # N_proposals

        idx_selected_proposals = torch.arange(
            len(score_per_proposal), device=score_per_proposal.device
        )[score_per_proposal > self.matching_config.confidence_thresh]
        if len(idx_selected_proposals) > self.matching_config.max_num_instances:
            logging.info(
                f"Selecting top {self.matching_config.max_num_instances} instances ..."
            )
            _, idx = torch.topk(
                score_per_proposal[idx_selected_proposals],
                k=self.matching_config.max_num_instances,
            )
            idx_selected_proposals = idx_selected_proposals[idx]
        pred_idx_objects = assigned_idx_object[idx_selected_proposals]
        pred_scores = score_per_proposal[idx_selected_proposals]
        pred_score_distribution = score_per_proposal_and_object[idx_selected_proposals]
        return (
            idx_selected_proposals,
            pred_idx_objects,
            pred_scores,
            pred_score_distribution,
        )
