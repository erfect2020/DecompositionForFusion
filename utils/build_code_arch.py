import option.options as option
from torch.utils.tensorboard import SummaryWriter
import socket
from datetime import datetime
import torch
from utils import util
import logging



def build_resume_state(train_args):
    opt = option.parse(train_args.opt, is_train=True)

    # loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['epoch'])  # check resume options
    else:
        resume_state = None


    if resume_state is None:
        util.mkdir_and_rename(opt['path']['experiments_root'])  # rename experiment folder if exists
        util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                     and 'pretrain_model' not in key and 'resume' not in key))

    return opt, resume_state

def build_logger(opt):
    util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                      screen=True, tofile=True)
    util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                      screen=True, tofile=True)

    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))

    # tensorboard logger
    if opt['use_tb_logger'] and 'debug' not in opt['name']:
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        CURRENT_DATETIME_HOSTNAME = '/' + current_time + '_' + socket.gethostname()
        tb_logger = SummaryWriter(log_dir='./tb_logger/' + opt['name'] + CURRENT_DATETIME_HOSTNAME)

    torch.backends.cudnn.deterministic = True
    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    seed = opt['train']['manual_seed']
    util.set_random_seed(seed)
    torch.backends.cudnn.benchmark = True

    return opt, logger, tb_logger

