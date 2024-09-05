# ???: Do we need this file?

import torch

from src.models import dropout
from src.models import ensemble
from src.models import ensemble_ssn
from src.models import lsn
from src.models import ssn_dropout
from src.models import nnj_unet
from src.models.vimh import UNet_Ensemble

# flipout imports
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
from src.models.unet_flipout import FlipoutUnet
from src.models.ssn_flipout import StochasticUnetFlippy


def create_model(config, is_test=False):
    model_name = config["architecture"].lower()

    model = None
    loss_fn = None

    if "ssn_flipout" == model_name:
        const_bnn_prior_parameters = {
            "prior_mu": 0.0,
            "prior_sigma": 1.0,
            "posterior_mu_init": 0.0,
            "posterior_rho_init": -3.0,
            "type": "Flipout",  # Flipout or Reparameterization
            "moped_enable": False,  # True to initialize mu/sigma from the pretrained dnn weights
            "moped_delta": 0.5,
        }
        if is_test:
            model = StochasticUnetFlippy(
                model_name,
                img_size=config["image_size"],
                diagonal=config["diagonal"],
                is_epistmic=True,
            )
            dnn_to_bnn(model.unet, const_bnn_prior_parameters)
        else:
            model = StochasticUnetFlippy(
                model_name, img_size=config["image_size"], diagonal=config["diagonal"]
            )
            dnn_to_bnn(model.unet, const_bnn_prior_parameters)
        if config["diagonal"]:
            loss_fn = lsn.StochasticSegmentationNetworkLossMCIntegral(
                num_mc_samples=20, pos_weight=True
            )
        else:
            loss_fn = lsn.StochasticSegmentationNetworkLossMCIntegral(num_mc_samples=20)

    elif "flipout" in model_name:
        model = FlipoutUnet(
            model_name,
            input_channels=3,
            num_classes=1,
            num_filters=[8, 16, 32, 64, 128],
        )
        const_bnn_prior_parameters = {
            "prior_mu": 0.0,
            "prior_sigma": 1.0,
            "posterior_mu_init": 0.0,
            "posterior_rho_init": -3.0,
            "type": "Flipout",  # Flipout or Reparameterization
            "moped_enable": False,  # True to initialize mu/sigma from the pretrained dnn weights
            "moped_delta": 0.5,
        }

        dnn_to_bnn(model, const_bnn_prior_parameters)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(4.0))

    elif "vimh" == model_name:
        model = UNet_Ensemble(
            num_models=config["ensemble_size"],
            mutliHead_layer="BDec2",
            num_in=3,
            num_classes=1,
        )
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(4.0))

    elif "ssn u-net" == model_name:
        if is_test:
            model = lsn.StochasticUnet(
                model_name,
                img_size=config["image_size"],
                diagonal=config["diagonal"],
                is_epistmic=True,
            )
        else:
            model = lsn.StochasticUnet(
                model_name, img_size=config["image_size"], diagonal=config["diagonal"]
            )
        if config["diagonal"]:
            loss_fn = lsn.StochasticSegmentationNetworkLossMCIntegral(
                num_mc_samples=20, pos_weight=True
            )
        else:
            loss_fn = lsn.StochasticSegmentationNetworkLossMCIntegral(num_mc_samples=20)
    elif "ssn u-net with dropout" == model_name:
        if is_test:
            model = ssn_dropout.StochasticUnet_with_Dropout(
                model_name, diagonal=config["diagonal"], is_epistmic=True
            )
        else:
            model = ssn_dropout.StochasticUnet_with_Dropout(
                model_name, diagonal=config["diagonal"]
            )
        if config["diagonal"]:
            loss_fn = lsn.StochasticSegmentationNetworkLossMCIntegral(
                num_mc_samples=20, pos_weight=True
            )
        else:
            loss_fn = lsn.StochasticSegmentationNetworkLossMCIntegral(num_mc_samples=20)
    elif "ensemble ssn" == model_name:

        model = ensemble_ssn.EnsembleSSN(model_name, config["ensemble_size"], 3, 1)
        if config["diagonal"]:
            loss_fn = lsn.StochasticSegmentationNetworkLossMCIntegral(
                num_mc_samples=20, pos_weight=True
            )
        else:
            loss_fn = lsn.StochasticSegmentationNetworkLossMCIntegral(num_mc_samples=20)
    elif "ensemble" in model_name:
        model = ensemble.Ensemble(
            model_name, config["ensemble_size"], 3, 1, [8, 16, 32, 64, 128]
        )
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(4.0))
    elif "dropout u-net" == model_name:
        model = dropout.DropoutUnet(model_name, 3, 1, [8, 16, 32, 64, 128])
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(8.0))
    elif "deterministic u-net" == model_name:
        model = nnj_unet.UNet_stochman_64(model_name, img_size=config["image_size"])
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(4.0))
    return model, loss_fn
