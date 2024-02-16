import torch
from tqdm.notebook import tqdm
from training import plot_losses


def distill_training_epoch(student, teacher, optimizer, criterion, train_loader, tqdm_desc):
    train_loss, train_accuracy = 0.0, 0.0
    student.train()
    device = next(student.parameters()).device

    for images, labels in tqdm(train_loader, desc=tqdm_desc):
        images = images.to(device)  # images: batch_size x num_channels x height x width
        labels = labels.to(device)  # labels: batch_size

        optimizer.zero_grad()
        student_logits = student(images)  # logits: batch_size x num_classes
        with torch.no_grad():
            teacher_logits = teacher(images)

        loss = criterion(student_logits, teacher_logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.shape[0]
        train_accuracy += (student_logits.argmax(dim=1) == labels).sum().item()
    
    train_loss /= len(train_loader.dataset)
    train_accuracy /= len(train_loader.dataset)
    return train_loss, train_accuracy


@torch.no_grad()
def distill_validation_epoch(student, teacher, criterion, test_loader, tqdm_desc):
    test_loss, test_accuracy = 0.0, 0.0
    student.eval()
    device = next(student.parameters()).device

    for images, labels in tqdm(test_loader, desc=tqdm_desc):
        images = images.to(device)  # images: batch_size x num_channels x height x width
        labels = labels.to(device)  # labels: batch_size

        student_logits = student(images)  # logits: batch_size x num_classes
        teacher_logits = teacher(images)
        loss = criterion(student_logits, teacher_logits, labels)

        test_loss += loss.item() * images.shape[0]
        test_accuracy += (student_logits.argmax(dim=1) == labels).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy /= len(test_loader.dataset)
    return test_loss, test_accuracy


def distill_train(student, teacher, optimizer, scheduler, criterion, train_loader, test_loader, num_epochs):
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    teacher.eval()

    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = distill_training_epoch(
            student, teacher, optimizer, criterion, train_loader,
            tqdm_desc=f'Training {epoch}/{num_epochs}'
        )
        test_loss, test_accuracy = distill_validation_epoch(
            student, teacher, criterion, test_loader,
            tqdm_desc=f'Validating {epoch}/{num_epochs}'
        )

        if scheduler is not None:
            scheduler.step()

        train_losses += [train_loss]
        train_accuracies += [train_accuracy]
        test_losses += [test_loss]
        test_accuracies += [test_accuracy]
        plot_losses(train_losses, test_losses, train_accuracies, test_accuracies)

    return train_losses, test_losses, train_accuracies, test_accuracies