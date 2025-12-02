""" Wandb配置脚本
1. 在训练脚本中导入：from wandb_config import init_wandb, log_training_metrics, log_validation_metrics
2. 训练开始前调用 init_wandb() 初始化
3. 训练循环中调用 log_training_metrics() 记录训练指标
4. 验证时调用 log_validation_metrics() 记录验证指标
5. 训练结束后调用 finish_wandb() 结束记录 """

import logging
import time
from typing import Optional, Dict, Any


def init_wandb(
    args=None,
    model=None,
    project: str = None,
    entity: str = None,
    run_name: str = None,
    config: Dict = None,
    watch_model: bool = False, # 是否监控模型梯度（默认False）
    watch_log_freq: int = 100,
    log_config: bool = True # 是否记录训练配置（默认True，设为False则不记录超参数）
) -> Optional[Any]:

    use_wandb = getattr(args, 'use_wandb', False) if args is not None else (project is not None)
    if not use_wandb:
        logging.info("Wandb未启用")
        return None
    
    try:
        import wandb
        
        if not log_config:
            config = {}
        elif config is None and args is not None:
            config = _extract_config_from_args(args)
        elif config is None:
            config = {}
        
        if args is not None:
            project = project or getattr(args, 'wandb_project', 'ml-training')
            entity = entity or getattr(args, 'wandb_entity', None)
            run_name = run_name or getattr(args, 'wandb_run_name', None) or getattr(args, 'exp_name', None)
        
        wandb_run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=config,
            resume='allow',
        )
        
        # 记录训练开始时间
        wandb_run._start_time = time.time()
        
        logging.info(f"Wandb初始化成功: {project}/{run_name}")
        logging.info(f"Wandb运行链接: {wandb_run.get_url()}")
        
        if model is not None and watch_model:
            try:
                wandb.watch(model, log='gradients', log_freq=watch_log_freq)
                logging.info(f"Wandb模型监控已启用（梯度记录频率: {watch_log_freq}步）")
            except Exception as e:
                logging.warning(f'wandb.watch 失败: {e}')
        
        return wandb_run
        
    except ImportError:
        logging.error("未安装wandb库，请运行: pip install wandb")
        return None
    except Exception as e:
        logging.warning(f"未能初始化 wandb（将继续不记录到 wandb）: {e}")
        return None


def _extract_config_from_args(args) -> Dict:
    # 从args对象中自动提取训练配置参数
    config = {}
    
    # 模型参数
    model_params = [
        'vocab_size',  
        'context_length',
        'd_model',
        'num_layers',
        'num_heads',
        'd_ff'
    ]
    for param in model_params:
        if hasattr(args, param):
            config[param] = getattr(args, param)
    
    # 训练参数
    training_params = [
        'batch_size',
        'learning_rate',
        'max_iters',
        'warmup_iters',
        'weight_decay',
        'grad_clip'
    ]
    for param in training_params:
        if hasattr(args, param):
            config[param] = getattr(args, param)
    
    # 优化器
    if hasattr(args, 'optimizer'):
        config['optimizer'] = args.optimizer
    
    # 设备和精度
    if hasattr(args, 'device'):
        config['device'] = args.device
    if hasattr(args, 'dtype'):
        config['dtype'] = args.dtype
    
    # 验证配置
    if hasattr(args, 'eval_interval'):
        config['eval_interval'] = args.eval_interval
    if hasattr(args, 'num_eval_batches'):
        config['num_eval_batches'] = args.num_eval_batches
    
    return config


def log_training_metrics(
    wandb_run,
    iteration: int,
    loss: float,
    learning_rate: float,
    elapsed_time: Optional[float] = None,
    **kwargs # 其他自定义指标（如cpu_percent, gpu_mem等）
):
    if wandb_run is None:
        return
    
    try:
        import wandb
        
        log_data = {
            'train/loss': loss,
            'train/lr': learning_rate,
            'train/iteration': iteration,
        }
        
        # 自动计算elapsed_time
        if elapsed_time is None and hasattr(wandb_run, '_start_time'):
            elapsed_time = time.time() - wandb_run._start_time
        
        if elapsed_time is not None:
            log_data['train/loss_vs_time'] = loss
            log_data['time/elapsed_hours'] = elapsed_time / 3600
            log_data['time/elapsed_seconds'] = elapsed_time
        
        log_data.update(kwargs)
        wandb.log(log_data, step=iteration)
        
    except Exception as e:
        logging.warning(f"记录训练指标到wandb失败: {e}")


def log_validation_metrics(
    wandb_run,
    iteration: int,
    val_loss: float,
    elapsed_time: Optional[float] = None,
    train_losses: list = None,
    val_losses: list = None,
    train_iterations: list = None,
    val_iterations: list = None,
    **kwargs
):
    if wandb_run is None:
        return
    
    try:
        import wandb
        
        log_data = {
            'val/loss': val_loss,
            'val/iteration': iteration,
        }
        
        if elapsed_time is None and hasattr(wandb_run, '_start_time'):
            elapsed_time = time.time() - wandb_run._start_time
        
        if elapsed_time is not None:
            log_data['val/loss_vs_time'] = val_loss
            log_data['time/elapsed_hours'] = elapsed_time / 3600
            log_data['time/elapsed_seconds'] = elapsed_time
        
        # 自动绘制并上传loss曲线图
        if train_losses is not None and val_losses is not None:
            try:
                import matplotlib.pyplot as plt
                import io
                from PIL import Image
                
                # 图1: X轴是迭代次数
                plt.figure(figsize=(10, 6))
                if train_iterations is not None:
                    plt.plot(train_iterations, train_losses, label='Train Loss', alpha=0.6)
                else:
                    plt.plot(train_losses, label='Train Loss', alpha=0.6)
                
                if val_iterations is not None:
                    plt.plot(val_iterations, val_losses, label='Validation Loss', alpha=0.6)
                else:
                    plt.plot(val_losses, label='Validation Loss', alpha=0.6)
                
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.title('Training Loss Curves')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                log_data['plots/loss_curves'] = wandb.Image(Image.open(buf))
                plt.close()
                
                # 图2: X轴是时间（小时）
                if hasattr(wandb_run, '_start_time'):
                    plt.figure(figsize=(10, 6))
                    
                    # 计算每个点的时间
                    current_time = time.time() - wandb_run._start_time
                    train_times = [current_time / 3600] * len(train_losses)
                    val_times = [current_time / 3600] * len(val_losses)
                    
                    if train_iterations is not None:
                        plt.plot(train_times, train_losses, label='Train Loss', alpha=0.6)
                    if val_iterations is not None:
                        plt.plot(val_times, val_losses, label='Validation Loss', alpha=0.6)
                    
                    plt.xlabel('Time (hours)')
                    plt.ylabel('Loss')
                    plt.title('Training Loss vs Time')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    buf_time = io.BytesIO()
                    plt.savefig(buf_time, format='png', dpi=100, bbox_inches='tight')
                    buf_time.seek(0)
                    log_data['plots/loss_vs_time'] = wandb.Image(Image.open(buf_time))
                    plt.close()
                
            except Exception as e:
                logging.warning(f"绘制loss曲线失败: {e}")
        
        log_data.update(kwargs)
        wandb.log(log_data, step=iteration)
        
    except Exception as e:
        logging.warning(f"记录验证指标到wandb失败: {e}")


def finish_wandb(wandb_run):
    if wandb_run is None:
        return
    
    try:
        import wandb
        wandb.finish()
        logging.info("Wandb运行已结束")
    except Exception as e:
        logging.warning(f"结束wandb运行失败: {e}")


def add_wandb_args(parser):
    parser.add_argument('--use_wandb', action='store_true',
                       help='启用 Weights & Biases 日志')
    parser.add_argument('--wandb_project', type=str, default='transformer',
                       help='wandb 项目名')
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='wandb 团队/用户名')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='wandb 运行名（可选）')
    parser.add_argument('--eval_interval', type=int, default=500,
                       help='验证间隔（每N次迭代验证一次）')
    parser.add_argument('--num_eval_batches', type=int, default=50,
                       help='验证时采样的batch数量')
    
    return parser


if __name__ == '__main__':
    print("这是一个wandb配置模块，请在你的训练脚本中导入使用")
