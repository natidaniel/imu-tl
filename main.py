"""
IMU-TLcdegn\
Entry point for training and testing IMU architectures for activity recognition with transfer learning
"""
import argparse
import torch
import numpy as np
import json
import logging
import matplotlib.pyplot as plt
from util import utils
from os.path import join
from models.IMUTransformerEncoder import IMUTransformerEncoder
from models.IMULSTM import IMULSTM
from models.IMUConv import IMUConv
from models.IMUResnet import IMUResnet
from util.IMUDataset import IMUDataset, split_train_val
from sklearn.metrics import confusion_matrix
from util.tsne import plot_tsne
from util.perutils import tic, toc


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("mode", help="train, transfer or test")
    arg_parser.add_argument("architecture", help="imu-cnn, imu-transformer, imu-lstm, imu-resnet")
    arg_parser.add_argument("imu_dataset_file", help="path to a file mapping imu samples to labels")
    arg_parser.add_argument("config_file",
                            help="path to config file (see examples in configs)")
    arg_parser.add_argument("--pretrained_path",
                            help="path to a pre-trained model")
    arg_parser.add_argument("--finetune", help="if set, only the classifier head will be trained",
                            action='store_true', default=False)
    arg_parser.add_argument('--val', action='store_true',
                            help='perform validation at the end of training')
    arg_parser.add_argument('--no_plot', dest='plot', action='store_false',
                        help='disable plotting')
    arg_parser.add_argument('--no_plot_tsne', dest='plot_tsne', action='store_false',
                        help='disable plotting tsne vis')


    args = arg_parser.parse_args()
    utils.init_logger()

    # Record execution details
    logging.info("Start learning-based inertial-based activity recognition".format(args.mode))
    logging.info("Using imu dataset file: {}".format(args.imu_dataset_file))

    # Read configuration
    with open('configs/example_config.json', "r") as read_file:
        config = json.load(read_file)
    logging.info("Running with configuration:\n{}".format(
        '\n'.join(["\t{}: {}".format(k, v) for k, v in config.items()])))

    # Set the seeds and the device
    use_cuda = torch.cuda.is_available()
    device_id = 'cpu'
    torch_seed = 0
    numpy_seed = 2
    torch.manual_seed(torch_seed)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device_id = config.get('device_id')
    np.random.seed(numpy_seed)
    device = torch.device(device_id)

    # Create the dataset and infer number of classes
    logging.info("Prepare dataset")
    window_shift = config.get("window_shift")
    window_size = config.get("window_size")
    input_size = config.get("input_dim")
    data_header = config.get("is_data_header")
    use_gps = config.get("use_gps")

    dataset = IMUDataset(args.imu_dataset_file, window_size, input_size, window_shift, header=data_header, use_gps=use_gps)
    if args.val:
        print("Splitting to train and validation")
        train_indices, val_indices = split_train_val(dataset, p=0.1)
        print("Prepare train set")
        dataset = IMUDataset(args.imu_dataset_file, window_size, input_size, window_shift, header=data_header,
                             selected_indices=train_indices, use_gps=use_gps)
        print("Prepare validation set")
        val_dataset = IMUDataset(args.imu_dataset_file, window_size, input_size, window_shift, header=data_header,
                             selected_indices=val_indices, use_gps=use_gps)


    config['num_classes'] = dataset.n_classes

    if args.architecture=='imu-cnn':
        model = IMUConv(config).to(device)
    elif args.architecture=='imu-transformer':
        model = IMUTransformerEncoder(config).to(device)
    elif args.architecture=='imu-lstm':
        model = IMULSTM(config).to(device)
    elif args.architecture=='imu-resnet':
        model = IMUResnet(config).to(device)
        #model = utils.to_cuda(model)
    else:
        raise NotImplementedError("{} architecture not supported".format(args.architetcure))

    # Log number of trainable model params
    logging.info("Train model: {}, Trainable params: {}".format(args.architecture, sum(
        [param.nelement() for param in model.parameters()])))

    pretrained_state_dict = None
    if args.pretrained_path:
        pretrained_state_dict = torch.load(args.pretrained_path, map_location=device_id)

    mode = args.mode
    if mode == 'transfer':
        # Ensure fine-tuning given a pre-trained model
        mode = 'train'
        assert pretrained_state_dict is not None
        # Remove keys from reference (for loading)
        classifier_prefix = model.get_classifier_head_prefix(True)
        keys_to_remove = [name for name in pretrained_state_dict.keys() if name.startswith(classifier_prefix)]
        for key in keys_to_remove:
            del pretrained_state_dict[key]

        # Freeze if needed in the target
        if args.finetune:
            classifier_prefix = model.get_classifier_head_prefix()
            for name, parameter in model.named_parameters():
                if not name.startswith(classifier_prefix):
                    parameter.requires_grad_(False)
                    print("Freezing param: [{}]".format(name))

    # Load the model if available
    if pretrained_state_dict is not None:
        model.load_state_dict(pretrained_state_dict, strict=False)
        logging.info("Initializing from checkpoint: {}".format(args.pretrained_path))

    if mode == 'train':
        # Init train run time
        tic()

        # Set to train mode
        model.train()

        # Set the loss
        loss = torch.nn.NLLLoss()

        # Set the optimizer and scheduler
        optim = torch.optim.Adam(model.parameters(),
                                  lr=config.get('lr'),
                                  eps=config.get('eps'),
                                  weight_decay=config.get('weight_decay'))
        scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                    step_size=config.get('lr_scheduler_step_size'),
                                                    gamma=config.get('lr_scheduler_gamma'))

        # Set the dataset and data loader
        loader_params = {'batch_size': config.get('batch_size'),
                                  'shuffle': True,
                                  'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)
        logging.info("Data preparation completed")

        # Get training details
        n_freq_print = config.get("n_freq_print")
        n_freq_checkpoint = config.get("n_freq_checkpoint")
        n_epochs = config.get("n_epochs")

        # Train
        checkpoint_prefix = join(utils.create_output_dir('out'),utils.get_stamp_from_log())
        n_total_samples = 0.0
        loss_vals = []
        sample_count = []
        batch_res = []
        batch_label = []

        logging.info("Start training")
        for epoch in range(n_epochs):

            for batch_idx, minibatch in enumerate(dataloader):
                if args.architecture == 'imu-resnet':
                    minibatch["imu"] = utils.to_cuda(minibatch["imu"]).to(dtype=torch.float32)
                else:
                    minibatch["imu"] = minibatch["imu"].to(device).to(dtype=torch.float32)

                label = minibatch.get('label').to(device).to(dtype=torch.long)
                batch_size = label.shape[0]

                n_total_samples += batch_size

                # Zero the gradients
                optim.zero_grad()

                # Forward pass
                if args.architecture=='imu-lstm':
                    model.init_hidden(batch_size, device)
                if args.architecture=='imu-resnet':
                    res = model(minibatch["imu"])
                else:
                    res = model(minibatch)

                # Compute loss
                criterion = loss(res, label)

                # Cat train features for last epoch only
                if epoch+1 == n_epochs:
                    if batch_idx == 0:
                        batch_res = res
                        batch_label = label
                    else:
                        batch_res = torch.cat((batch_res, res), 0)
                        batch_label = torch.cat((batch_label, label), 0)

                # Collect for recoding and plotting
                batch_loss = criterion.item()
                loss_vals.append(batch_loss)
                sample_count.append(n_total_samples)

                # Back prop
                criterion.backward()
                optim.step()

                # Record loss on train set
                if batch_idx % n_freq_print == 0:
                    logging.info("[Batch-{}/Epoch-{}] batch loss: {:.3f}".format(
                                                                        batch_idx+1, epoch+1,
                                                                        batch_loss))
            # Save checkpoint
            if (epoch % n_freq_checkpoint) == 0 and epoch > 0:
                torch.save(model.state_dict(), checkpoint_prefix + '_checkpoint-{}.pth'.format(epoch))

            # Scheduler update
            scheduler.step()

        # Log training run time
        logging.info('Training completed with run time: {} seconds'.format(toc()))
        torch.save(model.state_dict(), checkpoint_prefix + '_final.pth'.format(epoch))

        # plot t-SNE features visualization
        if args.plot:
            if args.plot_tsne:
                tsne_fig_path = checkpoint_prefix + "_" + args.architecture + "_trained_tsne_fig.png"
                plot_tsne(batch_res, batch_label, dataset, tsne_fig_path, dim=2, perplexity=30.0, scale_data=True)

            # Plot the loss function
            loss_fig_path = checkpoint_prefix + "_loss_fig.png"
            utils.plot_loss_func(sample_count, loss_vals, loss_fig_path)

    if args.val or args.mode == 'test': # Test
        if args.val:
            dataset = val_dataset
            args.mode = 'validation'
        # Set to eval mode
        model.eval()
        loader_params = {'batch_size': 1,
                         'shuffle': False,
                         'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)
        logging.info("Data preparation completed")

        metric = []
        batch_res = []
        batch_label = []

        logging.info("Start inference")
        # Init test run time
        tic()
        accuracy_per_label = np.zeros(config.get("num_classes"))
        count_per_label = np.zeros(config.get("num_classes"))
        predicted = []
        ground_truth = []
        with torch.no_grad():
            for i, minibatch in enumerate(dataloader, 0):
                if args.architecture == 'imu-resnet':
                    minibatch["imu"] = utils.to_cuda(minibatch["imu"]).to(dtype=torch.float32)
                else:
                    minibatch["imu"] = minibatch["imu"].to(device).to(dtype=torch.float32)
                label = minibatch.get('label').to(device).to(dtype=torch.long)
                raw_label = minibatch.get('raw_label').to(device).to(dtype=torch.long)
                batch_size =label.shape[0]

                # Forward pass
                if args.architecture=='imu-lstm':
                    model.init_hidden(batch_size, device)

                if args.architecture=='imu-resnet':
                    res = model(minibatch["imu"])
                else:
                    res = model(minibatch)

                # Cat test features
                if i == 0:
                    batch_res = res
                    batch_label = label
                else:
                    batch_res = torch.cat((batch_res, res), 0)
                    batch_label = torch.cat((batch_label, label), 0)

                # Evaluate and append
                pred_label = torch.argmax(res)
                predicted.append(pred_label.cpu().numpy())
                ground_truth.append(label.item())
                curr_metric = (pred_label==label).to(torch.int)
                label_id = label.item()
                accuracy_per_label[label_id] += curr_metric.item()
                count_per_label[label_id] += 1
                metric.append(curr_metric.item())

        # Log test run time
        logging.info('Evaluation for {} dataset is completed with run time: {} seconds'.format(args.mode, toc()))

        # Record overall statistics
        stats_msg = "Performance on {} dataset (file: {})".format(args.mode, args.imu_dataset_file)
        confusion_mat = confusion_matrix(ground_truth, predicted, labels=list(range(config.get("num_classes"))))
        stats_msg = stats_msg + "\n\tAccuracy: {:.3f}".format(np.mean(metric))
        accuracies = []
        target_names = []
        for i in range(len(accuracy_per_label)):
                raw_label = dataset.sorted_unique_raw_labels[i]
                raw_label_str = dataset.label_dict.get(str(raw_label))
                target_names.append(raw_label_str)
                logging.info("Performance for class [{}]/ id [{}] - accuracy {:.3f}".format(raw_label_str, raw_label, accuracy_per_label[i]/count_per_label[i]))
                accuracies.append(accuracy_per_label[i]/count_per_label[i])
        logging.info(stats_msg)

        # Plot CM
        if args.plot:
            utils.plot_confusion_matrix(confusion_mat,
                                        target_names,
                                        font_size=22,
                                        out_dir=args.pretrained_path,
                                        cmap=plt.get_cmap('Greens'),
                                        normalize=True)

            # plot t-SNE features visualization
            if args.plot_tsne:
                tsne_fig_path = args.pretrained_path + "_" + args.architecture + "_test_tsne_fig.png"
                plot_tsne(batch_res, batch_label, dataset, tsne_fig_path, dim=2, perplexity=30.0, scale_data=True)

        # save dump
        dump_name = args.pretrained_path + "_test_results_dump"
        np.savez(dump_name, confusion_mat=confusion_mat, accuracies=accuracies,
                 count_per_label=count_per_label, total_acc=np.mean(metric),
                 predicted=predicted, ground_truth=ground_truth, config=config,
                 sorted_unique_raw_labels=dataset.sorted_unique_raw_labels, label_dict=dataset.label_dict)
        logging.info("results saved to: {}.npz".format(dump_name))


