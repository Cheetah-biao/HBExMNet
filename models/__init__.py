import logging


logger = logging.getLogger("base")


def create_model(opt):
    model_name = opt["model"]
    if model_name == "SingleModel":
        from .SingleModel import SingleModel as model_cls
    elif model_name == "DualstageModel":
        from .DualstageGanModel import DualstageGanModel as model_cls
    elif model_name == "Triplestage_v":
        from .triple_stage_v_model import Triple_stage_v_Model as model_cls
    else:
        raise NotImplementedError(f"Model [{model_name}] is not supported in this release.")

    model = model_cls(opt)
    logger.info("Model [%s] is created.", model.__class__.__name__)
    return model
