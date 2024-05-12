import torch
import logging
import argparse
import torchbearer
from torch import optim
from torchbearer import Trial
from torchbearer.callbacks.torch_scheduler import CosineAnnealingLR

from utils import get_data_loaders
from resnet_model import ResNet

logging.getLogger().setLevel(logging.DEBUG)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data",
        help="The path to the dataset. (default: data)",
    )
    parser.add_argument(
        "--model_spec",
        type=str,
        default="CS",
        help="The model specification. CS for Code-Spec and PS for Paper-Spec. For more details, refer to the report. (default: CS)",
        choices=["CS", "PS"],
    )
    parser.add_argument("--batch_size", type=int, default=32, help="The batch size. (default: 32)")
    parser.add_argument("--epochs", type=int, default=50, help="The number of epochs. (default: 50)")
    parser.add_argument(
        "--topk_operation",
        type=str,
        default="none",
        help="The top-k operation. There are three options: none, top_k, and top_k_mean_replace. (default: none)",
        choices=["none", "top_k", "top_k_mean_replace"],
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="resnet_model.pth",
        help="The path to save the model. (default: resnet_model.pth)",
    )
    args = parser.parse_args()

    top_k = None
    if args.topk_operation != "none":
        top_k = args.topk_operation

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get the data loaders
    trainloader, testloader, stylised_testloader = get_data_loaders(
        dataset_path=args.dataset_path, batch_size=args.batch_size
    )

    # The ResNet model
    resnet = ResNet(model_spec=args.model_spec, topk_operation=top_k, device=device)

    # The loss function, optimiser, and scheduler
    criterion = torch.nn.CrossEntropyLoss()
    if args.model_spec == "CS":
        optimiser = optim.SGD(resnet.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    else:
        optimiser = optim.SGD(resnet.parameters(), lr=0.1)
    scheduler = CosineAnnealingLR(T_max=args.epochs)

    # The trial
    trial = Trial(
        resnet,
        optimiser,
        criterion,
        metrics=["loss", "accuracy"],
        callbacks=[scheduler],
    ).to(device)
    trial.with_generators(trainloader, test_generator=testloader).run(epochs=args.epochs)
    trial.run(epochs=args.epochs)

    # Evaluate the model on the test dataset
    results = trial.evaluate(data_key=torchbearer.TEST_DATA, verbose=0)
    test_accuracy = results["test_acc"]
    logging.info("Test accuracy: {}".format(test_accuracy))

    # Evaluate the model on the stylised test dataset
    trial.with_test_generator(stylised_testloader)
    stylised_results = trial.evaluate(data_key=torchbearer.TEST_DATA, verbose=0)
    stylised_accuracy = stylised_results["test_acc"]
    logging.info("Stylised test accuracy: {}".format(stylised_accuracy))

    # Save the model
    torch.save(resnet.state_dict(), args.output_path)
    logging.info("Model saved at: {}".format(args.output_path))
