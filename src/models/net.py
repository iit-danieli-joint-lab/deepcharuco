from torch import optim, nn
import torch
import pytorch_lightning as pl


class dcModel(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """

    def __init__(self, n_ids):
        super(dcModel, self).__init__()

        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unpool = torch.nn.MaxUnpool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256
        det_h = 65
        self.reBn = True

        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(3, c1, kernel_size=3, stride=1, padding=1)
        self.bn1a = nn.BatchNorm2d(c1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.bn1b = nn.BatchNorm2d(c1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.bn2a = nn.BatchNorm2d(c2)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.bn2b = nn.BatchNorm2d(c2)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.bn3a = nn.BatchNorm2d(c3)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.bn3b = nn.BatchNorm2d(c3)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.bn4a = nn.BatchNorm2d(c4)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        self.bn4b = nn.BatchNorm2d(c4)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnPa = nn.BatchNorm2d(c5)
        self.convPb = torch.nn.Conv2d(c5, det_h, kernel_size=1, stride=1, padding=0)

        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnDa = nn.BatchNorm2d(c5)

        self.convDb = torch.nn.Conv2d(c5, n_ids + 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x, subpixel=False):
        """
        Input
          x: Image pytorch tensor shaped N x 1 x H x W.
        Output
          loc: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          ids: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """

        # Let's stick to this version: first BN, then relu
        x = self.relu(self.bn1a(self.conv1a(x)))
        conv1 = self.relu(self.bn1b(self.conv1b(x)))
        x, ind1 = self.pool(conv1)
        x = self.relu(self.bn2a(self.conv2a(x)))
        conv2 = self.relu(self.bn2b(self.conv2b(x)))
        x, ind2 = self.pool(conv2)
        x = self.relu(self.bn3a(self.conv3a(x)))
        conv3 = self.relu(self.bn3b(self.conv3b(x)))
        x, ind3 = self.pool(conv3)
        x = self.relu(self.bn4a(self.conv4a(x)))
        x = self.relu(self.bn4b(self.conv4b(x)))

        # Detector Head.
        cPa = self.relu(self.bnPa(self.convPa(x)))
        loc = self.convPb(cPa)  # NO activ
        # Descriptor Head.
        cDa = self.relu(self.bnDa(self.convDa(x)))
        ids = self.convDb(cDa)  # NO activ

        output = {'loc': loc, 'ids': ids}

        return output


def conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, stride=2),
        nn.ReLU(inplace=True)
    )


def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True)
    )


# define the LightningModule
class lModel(pl.LightningModule):
    def __init__(self, dcModel):
        super().__init__()
        self.model = dcModel

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, (loc, ids) = batch.values()
        loc_hat, ids_hat = self.model(x).values()

        loss_loc = nn.functional.cross_entropy(loc_hat, loc)
        loss_ids = nn.functional.cross_entropy(ids_hat, ids)

        self.log("val_loss_loc", loss_loc)
        self.log("val_loss_ids", loss_ids)
        return loss_loc, loss_ids

    def training_step(self, batch, batch_idx):
        x, (loc, ids) = batch.values()

        # print('BATCH shapes')
        # print(x.shape)
        # print(loc.shape, ids.shape)

        loc_hat, ids_hat = self.model(x).values()

        # print('Model out shapes')
        # print(loc_hat.shape, ids_hat.shape)

        loss_loc = nn.functional.cross_entropy(loc_hat, loc)
        loss_ids = nn.functional.cross_entropy(ids_hat, ids)

        self.log("train_loss_loc", loss_loc)
        self.log("train_loss_ids", loss_ids)
        return loss_loc + loss_ids

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = dcModel(n_ids=16)
    model = model.to(device)

    from torchinfo import summary
    summary(model, input_size=(1, 3, 240, 320))
