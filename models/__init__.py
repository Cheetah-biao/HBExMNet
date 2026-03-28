import logging

logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']
    if model == 'SingleModel':
        from .SingleModel import SingleModel as M
    elif model == 'Single_v':
        from .single_v_model import ImageRestorationModel as M
    elif model == 'GanModel':
        from .GanModel import GanModel as M
    elif model == 'DualstageModel':
        from .DualstageGanModel import DualstageGanModel as M
    elif model == 'Dualstage_v':
        from .dual_stage_v_model import Dual_stage_v_Model as M
    elif model == 'Triplestage_v':
        from .triple_stage_v_model import Triple_stage_v_Model as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
