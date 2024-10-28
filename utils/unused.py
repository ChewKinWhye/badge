def train_prune(self, X_query, Y_query, P_query, X_val, Y_val, P_val, verbose=True):
    # Freeze all weight paramters since we are only updating the masks
    for name, param in self.clf.named_parameters():
        if "mask_weight" in name or "bn" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    for name, buff in self.clf.named_buffers():
        if "mask" in name:
            buff.fill_(True)

    optimizer = optim.Adam(self.clf.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

    # Obtain train and validation dataset and loader, dataset to train the mask is simply the queried points
    loader_tr = DataLoader(
        self.handler(X_query, torch.Tensor(Y_query).long(), torch.Tensor(P_query).long(), isTrain=True),
        shuffle=True, batch_size=self.args.batch_size)
    loader_val = DataLoader(self.handler(X_val, torch.Tensor(Y_val).long(), torch.Tensor(P_val).long(), isTrain=False),
                            shuffle=False, batch_size=self.args.batch_size)

    criterion = torch.nn.CrossEntropyLoss()

    # --- Train Start ---
    best_val_avg_acc, best_epoch = -1, None
    for epoch in range(self.num_epochs):
        self.clf.train()
        # Track metrics
        ce_loss_meter, train_minority_acc, train_majority_acc, train_avg_acc = AverageMeter(), AverageMeter(), \
                                                                               AverageMeter(), AverageMeter()
        start = time.time()
        for batch in tqdm.tqdm(loader_tr, disable=True):
            x, y, p, idxs = batch
            x, y, p, idxs = x.cuda(), y.cuda(), p.cuda(), idxs.cuda()
            optimizer.zero_grad()

            # Cross Entropy Loss
            logits = self.clf(
                x)  # Set temperature as 1 allows for soft masking. For binary mask, temperature should increase with epoch
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            # Monitor training stats
            ce_loss_meter.update(torch.mean(loss).detach().item(), x.size(0))
            train_avg_acc, train_minority_acc, train_majority_acc = update_meter(train_avg_acc, train_minority_acc,
                                                                                 train_majority_acc,
                                                                                 logits.detach(), y, p)

        self.clf.eval()
        val_minority_acc, val_majority_acc, val_avg_acc = self.evaluate_model(loader_val)
        # Save best model based on worst group accuracy
        if val_avg_acc > best_val_avg_acc:
            torch.save(self.clf.state_dict(), os.path.join(self.args.save_dir, "ckpt.pt"))
            best_val_avg_acc = val_avg_acc
            best_epoch = epoch
        # Print stats
        if verbose:
            print(f"Epoch {epoch} Loss: {ce_loss_meter.avg:.3f} Time Taken: {time.time() - start:.3f}")
            print(
                f"Train Average Accuracy: {train_avg_acc.avg:.3f} Train Majority Accuracy: {train_majority_acc.avg:.3f} "
                f"Train Minority Accuracy: {train_minority_acc.avg:.3f}")
            print(f"Val Average Accuracy: {val_avg_acc:.3f} Val Majority Accuracy: {val_majority_acc:.3f} "
                  f"Val Minority Accuracy: {val_minority_acc:.3f}")
    # --- Train End ---
    print(f'Best validation accuracy: {best_val_avg_acc:.3f} at epoch {best_epoch}')
    state_dict = torch.load(os.path.join(self.args.save_dir, "ckpt.pt"))
    self.clf = get_model(self.args.pretrained, self.args.architecture, self.num_classes).cuda()
    self.clf.load_state_dict(state_dict, strict=False)
    return state_dict

class CustomConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
        padding_mode='zeros',device=None, dtype=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
        padding_mode,device, dtype)
        self.mask_weight = nn.Parameter(torch.zeros_like(self.weight))
        self.mask_weight.requires_grad = False
        self.register_buffer('mask', torch.tensor(False, dtype=torch.bool))
    def forward(self, input):
        if self.mask:
            return self._conv_forward(input, self.weight*sigmoid(self.mask_weight), self.bias)
        else:
            return self._conv_forward(input, self.weight, self.bias)
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def meta_forward(self, input, fast_weights, name):
        return self._conv_forward(input, fast_weights[name+".weight"], self.bias) # We just use self.bias since all the convolutions have bias=False

class CustomFC(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.mask_weight = nn.Parameter(torch.zeros_like(self.weight))
        self.mask_weight.requires_grad = False
        self.register_buffer('mask', torch.tensor(False, dtype=torch.bool))
    def forward(self, input):
        if self.mask:
            return F.linear(input, self.weight*sigmoid(self.mask_weight), self.bias)
        else:
            return F.linear(input, self.weight, self.bias)
    def meta_forward(self, input, fast_weights, name):
        return F.linear(input, fast_weights[name+".weight"], fast_weights[name+".bias"])
