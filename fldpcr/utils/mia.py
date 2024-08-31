##########################################################################
# Copyright 2024 Jianping Cai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################
# The implements of Membership Inference Attack
##########################################################################
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
import copy

from tqdm import tqdm

class ShadowAttackModel(nn.Module):
    def __init__(self, class_num):
        super(ShadowAttackModel, self).__init__()
        self.Output_Component = nn.Sequential(
            nn.Linear(class_num, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        self.Prediction_Component = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        self.Encoder_Component = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, output, prediction):
        Output_Component_result = self.Output_Component(output)
        Prediction_Component_result = self.Prediction_Component(prediction)

        final_inputs = torch.cat((Output_Component_result, Prediction_Component_result), 1)
        final_result = self.Encoder_Component(final_inputs)

        return final_result


class shadow():
    def __init__(self, trainloader, testloader, model, device):
        self.device = device
        self.model = model.to(self.device)
        self.trainloader = trainloader
        self.testloader = testloader

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(model.parameters(), lr=1e-1)

    def train(self):
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return 1. * correct / total

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in self.testloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)

                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return 1. * correct / total


def train_shadow_model(shadow_model, train_loader, test_loader, device, shadow_model_epoch=50):
    model = shadow(train_loader, test_loader, shadow_model, device)
    acc_train = 0
    acc_test = 0

    print("Training Shadow model!!!")
    for i in tqdm(range(shadow_model_epoch)):
        acc_train = model.train()
        acc_test = model.test()
        overfitting = round(acc_train - acc_test, 6)

    print("Shadow model training finished!!!")

    return model.model, acc_train, acc_test, overfitting


def get_attack_dataset_with_shadow(target_train, target_test, shadow_train, shadow_test, batch_size):
    mem_train, nonmem_train, mem_test, nonmem_test = list(shadow_train), list(shadow_test), list(target_train), list(
        target_test)

    for i in range(len(mem_train)):
        mem_train[i] = mem_train[i] + (1,)
    for i in range(len(nonmem_train)):
        nonmem_train[i] = nonmem_train[i] + (0,)
    for i in range(len(nonmem_test)):
        nonmem_test[i] = nonmem_test[i] + (0,)
    for i in range(len(mem_test)):
        mem_test[i] = (mem_test[i][0], int(mem_test[i][1]))
        mem_test[i] = mem_test[i] + (1,)

    train_length = min(len(mem_train), len(nonmem_train))
    test_length = min(len(mem_test), len(nonmem_test))

    mem_train, _ = torch.utils.data.random_split(mem_train, [train_length, len(mem_train) - train_length])
    non_mem_train, _ = torch.utils.data.random_split(nonmem_train, [train_length, len(nonmem_train) - train_length])
    mem_test, _ = torch.utils.data.random_split(mem_test, [test_length, len(mem_test) - test_length])
    non_mem_test, _ = torch.utils.data.random_split(nonmem_test, [test_length, len(nonmem_test) - test_length])

    attack_train = mem_train + non_mem_train
    attack_test = mem_test + non_mem_test

    attack_trainloader = torch.utils.data.DataLoader(
        attack_train, batch_size=batch_size, shuffle=True)
    attack_testloader = torch.utils.data.DataLoader(
        attack_test, batch_size=batch_size, shuffle=True)

    return attack_trainloader, attack_testloader


def get_attack_dataset_with_shadow_for_train(shadow_train, shadow_test, batch_size):
    mem_train, nonmem_train = list(shadow_train), list(shadow_test)

    for i in range(len(mem_train)):
        mem_train[i] = mem_train[i] + (1,)
    for i in range(len(nonmem_train)):
        nonmem_train[i] = nonmem_train[i] + (0,)

    train_length = min(len(mem_train), len(nonmem_train))
    mem_train, _ = torch.utils.data.random_split(mem_train, [train_length, len(mem_train) - train_length])
    non_mem_train, _ = torch.utils.data.random_split(nonmem_train, [train_length, len(nonmem_train) - train_length])
    attack_train = mem_train + non_mem_train
    attack_trainloader = torch.utils.data.DataLoader(
        attack_train, batch_size=batch_size, shuffle=True)
    return attack_trainloader


def get_attack_dataset_with_shadow_for_test(target_train, target_test, batch_size):
    mem_test, nonmem_test = list(target_train), list(target_test)

    for i in range(len(nonmem_test)):
        nonmem_test[i] = nonmem_test[i] + (0,)
    for i in range(len(mem_test)):
        mem_test[i] = (mem_test[i][0], int(mem_test[i][1]))
        mem_test[i] = mem_test[i] + (1,)
    test_length = min(len(mem_test), len(nonmem_test))
    mem_test, _ = torch.utils.data.random_split(mem_test, [test_length, len(mem_test) - test_length])
    non_mem_test, _ = torch.utils.data.random_split(nonmem_test, [test_length, len(nonmem_test) - test_length])
    attack_test = mem_test + non_mem_test
    attack_testloader = torch.utils.data.DataLoader(
        attack_test, batch_size=batch_size, shuffle=False)
    return attack_testloader


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


class MyDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __len__(self):
        return self.data_tensor[0].size(0)

    def __getitem__(self, idx):
        return (
            self.data_tensor[0][idx],
            self.data_tensor[1][idx],
            self.data_tensor[2][idx]
        )


class attack_for_blackbox:

    def __init__(self, attack_train_loader,
                 shadow_model, attack_model, device):
        self.device = device

        self.shadow_model = shadow_model.to(self.device)

        self.shadow_model.eval()

        self.attack_train_loader = attack_train_loader

        self.attack_model = attack_model.to(self.device)
        self.attack_model.apply(weights_init)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.attack_model.parameters(), lr=1e-5)

    def _get_data(self, model, inputs, targets):
        result = model(inputs)

        output, _ = torch.sort(result, descending=True)
        _, predicts = result.max(1)

        prediction = predicts.eq(targets).float()
        return output, prediction.unsqueeze(-1)

    def prepare_dataset(self):

        outputs_list = []
        predictions_list = []
        members_list = []

        for i, (inputs, targets, members) in enumerate(self.attack_train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            output, prediction = self._get_data(self.shadow_model, inputs, targets)
            members = members.to(self.device)
            outputs_list.append(output.detach())
            predictions_list.append(prediction.detach())
            members_list.append(members.detach())

        attack_train_data_tensor = (
            torch.cat(outputs_list, dim=0),
            torch.cat(predictions_list, dim=0),
            torch.cat(members_list, dim=0)
        )
        dataset = MyDataset(attack_train_data_tensor)
        batch_size = 30
        self.attack_train_data = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        print("Finished Train Dataset")

    def train(self, epoch):
        self.attack_model.train()
        batch_idx = 1
        train_loss = 0
        correct = 0
        total = 0

        final_train_gndtrth = []
        final_train_predict = []
        final_train_probabe = []

        final_result = []
        for output, prediction, members in self.attack_train_data:
            output, prediction, members = output.to(self.device), prediction.to(self.device), members.to(self.device)
            results = self.attack_model(output, prediction)
            results = F.softmax(results, dim=1)
            losses = self.criterion(results, members)
            losses.backward()
            self.optimizer.step()
            train_loss += losses.item()
            _, predicted = results.max(1)
            total += members.size(0)
            correct += predicted.eq(members).sum().item()
            if epoch:
                final_train_gndtrth.append(members)
                final_train_predict.append(predicted)
                final_train_probabe.append(results[:, 1])
            batch_idx += 1
        if epoch:
            print('Train Acc: %.3f%% (%d/%d) | Loss: %.3f' % (
                100. * correct / total, correct, total, 1. * train_loss / batch_idx))
            print("Saved Attack Train Ground Truth and Predict Sets")

        final_result.append(1. * correct / total)

        return final_result

    def test(self, target_model, attack_test_loader):

        target_model = target_model.to(self.device)
        target_model.eval()

        self.attack_model.eval()
        batch_idx = 1
        correct = 0
        total = 0

        final_test_gndtrth = []
        final_test_predict = []
        final_test_probabe = []

        final_result = []

        with torch.no_grad():
            for inputs, targets, members in attack_test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                output, prediction = self._get_data(target_model, inputs, targets)
                members = members.to(self.device)

                results = self.attack_model(output, prediction)
                _, predicted = results.max(1)
                total += members.size(0)
                correct += predicted.eq(members).sum().item()
                results = F.softmax(results, dim=1)

                final_test_gndtrth.append(members)
                final_test_predict.append(predicted)
                final_test_probabe.append(results[:, 1])

                batch_idx += 1


        print("Saved Attack Test Ground Truth and Predict Sets")
        final_result.append(1. * correct / total)
        print('Test Acc: %.3f%% (%d/%d)' % (100. * correct / (1.0 * total), correct, total))

        return final_result

    def judge(self, target_model, attack_test_loader, member_score, mia_momentum=0.0):
        member_score = [c * mia_momentum for c in member_score]
        target_model = target_model.to(self.device)
        target_model.eval()

        self.attack_model.eval()
        cnt_ind = 0
        with torch.no_grad():
            for inputs, targets, members in attack_test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                output, prediction = self._get_data(target_model, inputs, targets)
                results = self.attack_model(output, prediction)
                _, predicted = results.max(1)
                rr = predicted.cpu().numpy()
                rr = rr * 2 - 1
                member_score[cnt_ind:(cnt_ind + len(predicted))] += rr
                cnt_ind += len(predicted)

        return member_score

    def check(self, attack_test_loader, cnt):
        cnt = copy.deepcopy(cnt)
        cnt_ind = 0
        all_sample_cnt = 0
        right_sample_cnt = 0
        cnt = [(1 if p > 0 else (-1 if p < 0 else 0)) for p in cnt]
        for inputs, targets, members in attack_test_loader:
            predicted = cnt[cnt_ind:(cnt_ind + len(members))]
            mm = members.cpu().numpy()
            mm = 2 * mm - 1
            all_sample_cnt += sum(p != 0 for p in predicted)
            right_sample_cnt += sum(mm == predicted)
            cnt_ind += len(members)
        return right_sample_cnt / all_sample_cnt


# black shadow
def build_attack_model(attack_trainloader, shadow_model, attack_model, device, attack_model_epoch=50):
    attack = attack_for_blackbox(attack_trainloader, shadow_model, attack_model, device)
    attack.prepare_dataset()

    print("Building Attack Model.")
    for i in tqdm(range(attack_model_epoch)):
        flag = 1 if i == attack_model_epoch - 1 else 0
        res_train = attack.train(flag)

    print("Attack Model Built Successfully.")

    return attack, res_train


from torch.utils.data import DataLoader


class mia_helper:
    def __init__(self, mia_config, device):
        self.mia_config = mia_config
        self.device = device
        self.member_score = None

    def set_datasets(self, shadowTrainSet, shadowTestSet):
        self.shadowTrainSet = shadowTrainSet
        self.shadowTestSet = shadowTestSet

    def set_victim_datasets(self, memberSet, nonmemberSet, batch_size):
        self.memberSet = memberSet
        self.nonmemberSet = nonmemberSet
        self.victim_dataloader = get_attack_dataset_with_shadow_for_test(
            self.memberSet, self.nonmemberSet, batch_size)
        self.member_score = [0 for i in range(len(self.victim_dataloader.dataset))]

    def train_shadow_model(self, shadowModel0, train_batch_sizes, test_batch_sizes):
        shadowTrainLoader = DataLoader(self.shadowTrainSet, batch_size=train_batch_sizes, shuffle=False)
        shadowTestLoader = DataLoader(self.shadowTestSet, batch_size=test_batch_sizes, shuffle=False)
        self.shadowModel, acc_train, acc_test, overfitting = train_shadow_model(shadowModel0, shadowTrainLoader,
                                                                                shadowTestLoader, self.device,
                                                                                self.mia_config['shadow_model_epoch'])
        print(f'Train Acc of Shadow Model: {acc_train * 100:.3f}%')
        print(f'Test Acc of Shadow Model: {acc_test * 100:.3f}%')
        print(f'The overfitting rate is {overfitting * 100:.3f}%')

    def build_attack_model(self, num_classes, train_batch_size):
        attack_trainloader = get_attack_dataset_with_shadow_for_train(
            self.shadowTrainSet, self.shadowTestSet, train_batch_size)

        self.attack_model, res_train = build_attack_model(attack_trainloader, self.shadowModel,
                                                          ShadowAttackModel(num_classes),
                                                          self.device, self.mia_config['attack_model_epoch'])
        Acc = res_train[0]
        print(f'MIA Metrics ==> Train Acc: {Acc * 100:.3f}%')
        return self.attack_model

    def attack(self, victim_model):
        self.member_score = self.attack_model.judge(victim_model, self.victim_dataloader, self.member_score,
                                                    self.mia_config['mia_momentum'])

    def get_mia_acc(self):
        return self.attack_model.check(self.victim_dataloader, self.member_score)
