import net


def train(model, train_loader, optimizer, scheduler, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (ldr_images, true_hdr) in enumerate(train_loader):
            optimizer.zero_grad()
            pred_hdr, f2_2_aligned_1, f2_2_aligned_3, f2_2 = model(ldr_images)
            loss = net.compute_loss(pred_hdr, true_hdr, f2_2_aligned_1, f2_2_aligned_3, f2_2)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            if i % 10 == 9:
                # print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}],
                # Loss: {running_loss/10:.4f}')
                running_loss = 0.0