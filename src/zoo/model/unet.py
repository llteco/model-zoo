"""
Copyright (C) 2026 The ZOO Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import torch.nn as nn


class UNetSeeInDark(nn.Module):
    r"""The original U-Net architecture from "Learning to See In the Dark"."""

    def __init__(
        self,
        in_nc: int = 4,
        out_nc: int = 4,
        nf: int = 32,
        res: bool = False,
    ):
        super().__init__()
        self.res = res

        self.conv1_1 = nn.Conv2d(in_nc, nf, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = nn.Conv2d(nf, nf * 2, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(nf * 2, nf * 2, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3_1 = nn.Conv2d(nf * 2, nf * 4, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4_1 = nn.Conv2d(nf * 4, nf * 8, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(nf * 8, nf * 8, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5_1 = nn.Conv2d(nf * 8, nf * 16, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(nf * 16, nf * 16, kernel_size=3, stride=1, padding=1)

        self.upv6 = nn.ConvTranspose2d(nf * 16, nf * 8, 2, stride=2)
        self.conv6_1 = nn.Conv2d(nf * 16, nf * 8, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(nf * 8, nf * 8, kernel_size=3, stride=1, padding=1)

        self.upv7 = nn.ConvTranspose2d(nf * 8, nf * 4, 2, stride=2)
        self.conv7_1 = nn.Conv2d(nf * 8, nf * 4, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, padding=1)

        self.upv8 = nn.ConvTranspose2d(nf * 4, nf * 2, 2, stride=2)
        self.conv8_1 = nn.Conv2d(nf * 4, nf * 2, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(nf * 2, nf * 2, kernel_size=3, stride=1, padding=1)

        self.upv9 = nn.ConvTranspose2d(nf * 2, nf, 2, stride=2)
        self.conv9_1 = nn.Conv2d(nf * 2, nf, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)

        self.conv10_1 = nn.Conv2d(nf, out_nc, kernel_size=1, stride=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        conv1 = self.relu(self.conv1_1(x))
        conv1 = self.relu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)

        conv2 = self.relu(self.conv2_1(pool1))
        conv2 = self.relu(self.conv2_2(conv2))
        pool2 = self.pool2(conv2)

        conv3 = self.relu(self.conv3_1(pool2))
        conv3 = self.relu(self.conv3_2(conv3))
        pool3 = self.pool3(conv3)

        conv4 = self.relu(self.conv4_1(pool3))
        conv4 = self.relu(self.conv4_2(conv4))
        pool4 = self.pool4(conv4)

        conv5 = self.relu(self.conv5_1(pool4))
        conv5 = self.relu(self.conv5_2(conv5))

        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.relu(self.conv6_1(up6))
        conv6 = self.relu(self.conv6_2(conv6))

        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.relu(self.conv7_1(up7))
        conv7 = self.relu(self.conv7_2(conv7))

        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.relu(self.conv8_1(up8))
        conv8 = self.relu(self.conv8_2(conv8))

        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.relu(self.conv9_1(up9))
        conv9 = self.relu(self.conv9_2(conv9))

        conv10 = self.conv10_1(conv9)
        if self.res:
            out = conv10 + x
        else:
            out = conv10
        return out
