import torch
import argparse

from models.Model import set_seed, train_cross_entropy, train_cross_entropy_and_contrastive_center_loss


def main():
    parser = argparse.ArgumentParser(description='Train model')

    parser.add_argument('--n_speakers', action="store", dest="n_speakers", type=int,
                        help='the number of speakers wanted in the dataset, needs to be <= 1211')
    parser.add_argument('--dataset_dir', action="store", dest="dataset_dir", type=str,
                        help='path to the directory of the dataset')
    parser.add_argument('--checkpoint_dir', action="store", dest="checkpoint_dir", type=str,
                        help='path to save the checkpoints', default='.')

    parser.add_argument('--ccl_reg', action="store_true", dest="ccl_reg", default=False,
                        help='if True, then training with contrastive-center loss regularization')

    parser.add_argument('--batch_size', action="store", dest="batch_size", type=int, default=64)
    parser.add_argument('--n_epochs', action="store", dest="n_epochs", type=int, default=20)
    parser.add_argument('--num_workers', action="store", dest="num_workers", type=int, default=2)
    # For ResNet optimizer
    parser.add_argument('--resnet_betas', action="store", dest="betas", type=tuple, default=(0.9, 0.98),
                        help="Resnet's Adam optimizer betas")
    parser.add_argument('--resnet_epsilon', action="store", dest="epsilon", type=float, default=1e-9,
                        help="Resnet's Adam optimizer epsilon")
    parser.add_argument('--resnet_step_size', action="store", dest="step_size", type=int, default=7,
                        help="step size of Resnet's learning rate scheduler")
    parser.add_argument('--resnet_gamma', action="store", dest="gamma", type=int, default=0.1,
                        help="gamma of Resnet's learning rate scheduler")
    parser.add_argument('--resnet_lr', action="store", dest="resnet_lr", type=float, default=1e-4,
                        help="Resnet's Adam optimizer learning rate")
    # For Contrastive-center loss optimizer
    parser.add_argument('--ccl_optimizer', action="store", dest="ccl_optimizer", type=str, default='Adagrad',
                        help="contrastive-center loss optimizer name from torch.optim")
    parser.add_argument('--ccl_lr', action="store", dest="ccl_lr", type=float, default=0.001,
                        help="Resnet's Adam optimizer learning rate")
    parser.add_argument('--ccl_lambda', action="store", dest="ccl_lambda", type=float, default=1.0,
                        help="Resnet's Adam optimizer learning rate")

    args = parser.parse_args()

    set_seed(0)
    # device - cpu or gpu?
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.ccl_reg:
        train_cross_entropy_and_contrastive_center_loss(args.n_speakers, device, dataset_dir=args.dataset_dir,
                                                        checkpoint_path=args.checkpoint_dir, batch_size=args.batch_size,
                                                        num_epochs=args.n_epochs, num_workers=args.num_workers,
                                                        eps=args.epsilon, betas=args.betas, step_size=args.step_size,
                                                        gamma=args.gamma, lr=args.resnet_lr,
                                                        optimizer_name_ccl=args.ccl_optimizer, lr_ccl=args.ccl_lr,
                                                        lambda_c=args.ccl_lambda, download_dataset=False)
    else:
        train_cross_entropy(args.n_speakers, device, dataset_di=args.dataset_dir, checkpoint_path=args.checkpoint_dir,
                            batch_size=args.batch_size, num_epochs=args.n_epochs, num_workers=args.num_workers,
                            eps=args.epsilon, betas=args.betas, step_size=args.step_size, gamma=args.gamma,
                            lr=args.resnet_lr, download_dataset=False)

    return


if __name__ == '__main__':
    main()
