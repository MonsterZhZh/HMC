import scipy.io
import numpy as np
import torch
import torch.nn as nn

class HEXLoss(nn.Module):
    def __init__(self, cliques, stateSpace, numVar, cliqParents, childVariables, upPass, sumProduct,
                 upMsgTable, downMsgTable, variables, varTable, Eh):
        super(HEXLoss, self).__init__()
        self.cliques = cliques
        self.stateSpace = stateSpace
        self.numVar = numVar
        self.cliqParents = cliqParents
        self.childVariables = childVariables
        self.upPass = upPass
        self.sumProduct = sumProduct
        self.upMsgTable = upMsgTable
        self.downMsgTable = downMsgTable
        self.variables = variables
        self.varTable = varTable
        self.Eh = Eh

    def forward(self, fs, labels, device):
        # fs = self.normalization(fs, self.Eh)
        potentials = self.assignPotential(self.cliques, self.stateSpace, self.numVar, fs)
        messages = self.messagePassing(self.cliqParents, self.childVariables, self.upPass, self.sumProduct,
                                       potentials, self.upMsgTable, self.downMsgTable, device)
        pMargin, z = self.marginalProbability(self.variables, self.cliques, self.varTable,
                                              messages, potentials, device)

        loss = torch.zeros(fs.shape[0], dtype=torch.float64).to(device)
        for i in range(fs.shape[0]):
            label = labels[i].unsqueeze(0)
            selected_Pr = torch.gather(pMargin[:, i], 0, label)
            if selected_Pr == 0 or z[i] == 0 or torch.isinf(z[i]):
                continue
            loss[i] = -torch.log(selected_Pr)
        return torch.mean(loss)

    def inference(self, fs, device):
        with torch.no_grad():
            # fs = self.normalization(fs, self.Eh)
            potentials = self.assignPotential(self.cliques, self.stateSpace, self.numVar, fs)
            messages = self.messagePassing(self.cliqParents, self.childVariables, self.upPass, self.sumProduct,
                                           potentials, self.upMsgTable, self.downMsgTable, device)
            pMargin, z = self.marginalProbability(self.variables, self.cliques, self.varTable,
                                                  messages, potentials, device)
            return pMargin

    def assignPotential(self, cliques, stateSpace, numVar, fs):
        numC = np.size(cliques, 1)
        potentials = np.empty((numC, 1), dtype=object)
        eTable = fs.T.to(torch.float64) / numVar.to(torch.float64)

        for i in range(numC):
            vc = cliques[0, i]
            vs = stateSpace[i, 0]
            potential = torch.mm(vs.to(torch.float64), torch.index_select(eTable, 0, (vc - 1).squeeze()))
            potentials[i] = [torch.exp(potential)]

        return potentials

    def messagePassing(self, cliqParents, childVariables, upPass, sumProduct, potentials, upMsgTable, downMsgTable, device):
        numC = np.size(childVariables, 0)
        messages = np.empty((numC, 1), dtype=object)
        for i in range(numC):
            c = upPass[i, 0]
            nei = sumProduct[c - 1, 0]
            numState = np.size(nei, 1)
            vm = torch.zeros((np.size(nei, 0), np.size(nei, 1), potentials[0, 0].shape[1]), dtype=torch.float64).to(device)
            children = childVariables[c - 1, 0]
            for j in range(children.shape[0]):
                child = children[j, 0]
                lenTable = upMsgTable[child - 1, 0].shape[1]
                prodMsg = [[] for i in range(lenTable)]
                visited = torch.zeros((lenTable, 1), dtype=torch.bool).to(device)
                for m in range(numState):
                    sumMsg = 0
                    vNei = nei[j, m]
                    for n in range(vNei.shape[0]):
                        numVei = vNei[n, 0]
                        pNei = potentials[child - 1, 0][numVei - 1, :]
                        idx = torch.where(upMsgTable[child - 1, 0] == numVei)[1]
                        if not visited[idx, 0]:
                            # prod may differ between Matlab prod() and torch.prod()
                            prodMsg[idx] = torch.prod(messages[child - 1, 0][:-1, numVei - 1, :], 0)
                            visited[idx, 0] = True
                        sumMsg = sumMsg + pNei * prodMsg[idx]
                    vm[j, m, :] = sumMsg
            messages[c - 1] = [vm]

        for i in range(numC - 2, -1, -1):
            c = upPass[i, 0]
            nei = sumProduct[c - 1, 0]
            numState = np.size(nei, 1)
            vm = torch.zeros(messages[c - 1, 0].size(), dtype=torch.float64).to(device)
            vm[:-1, :, :] = messages[c - 1, 0][:-1, :, :]
            cParent = cliqParents[c - 1, 0]
            lenTable = downMsgTable[cParent - 1, 0].shape[1]
            prodMsg = [[] for i in range(lenTable)]
            visited = torch.zeros((lenTable, 1), dtype=torch.bool).to(device)
            for j in range(numState):
                sumMsg = 0
                vNei = nei[-1, j]
                for k in range(vNei.shape[0]):
                    numVei = vNei[k, 0]
                    pNei = potentials[cParent - 1, 0][numVei - 1, :]
                    idx = torch.where(downMsgTable[cParent - 1, 0] == numVei)[1]
                    if not visited[idx, 0]:
                        prodMsg[idx] = torch.prod(messages[cParent - 1, 0][:, numVei - 1, :][downMsgTable[c - 1, 1].squeeze()], 0)
                        visited[idx, 0] = True
                    sumMsg = sumMsg + pNei * prodMsg[idx]
                vm[-1, j, :] = sumMsg
            messages[c - 1] = [vm]
        return messages

    def marginalProbability(self, variables, cliques, varTable, messages, potentials, device):
        numV = np.size(variables)
        numC = np.size(cliques)
        cBelief = np.empty((numC, 1), dtype=object)
        for c in range(numC):
            pc = potentials[c, 0]
            mc = messages[c, 0]
            belief = pc * torch.prod(mc, 0)
            cBelief[c] = [belief]
        z = torch.sum(cBelief[0, 0], 0)
        pMargin = torch.zeros((numV, cBelief[0, 0].shape[1]), dtype=torch.float64).to(device)
        for v in range(1, numV + 1):
            c = variables[v - 1, 0][0, 0]
            var = varTable[c - 1, 0]
            belief = cBelief[c - 1, 0]
            pMargin[v - 1, :] = torch.sum(torch.index_select(belief, 0, (var[(cliques[0, c - 1] == v).cpu()][0] - 1).squeeze()), 0)
        pMargin = pMargin / z
        return pMargin, z

    def normalization(self, fs, Eh):
        normed_fs = fs.clone()
        all = torch.sum(Eh)
        belong = torch.sum(Eh, 1)
        belong = belong[belong > 0]
        ratios = all / belong.to(torch.float64)
        positions = torch.nonzero(Eh)
        for i in range(positions.shape[0]):
            root = positions[i, 0]
            leaf = positions[i, 1]
            ratio = ratios[root]
            normed_fs[:, leaf] = fs[:, leaf] * ratio
        return normed_fs


def convert_to_torch(ndarray, device):
    i, j = ndarray.shape
    if i == 1 and j > 1:
        for k in range(j):
            ndarray[0, k] = torch.from_numpy(ndarray[0, k].astype(np.int64)).to(torch.long).to(device)
    elif j == 1 and i > 1:
        for k in range(i):
            ndarray[k, 0] = torch.from_numpy(ndarray[k, 0].astype(np.int64)).to(torch.long).to(device)
    elif i > 1 and j == 2:
        # Special for downMsgTable
        for k in range(i):
            ndarray[k, 0] = torch.from_numpy(ndarray[k, 0].astype(np.int64)).to(torch.long).to(device)
            ndarray[k, 1] = ndarray[k, 1].astype(np.bool)
            ndarray[k, 1] = torch.from_numpy(ndarray[k, 1]).to(torch.bool).to(device)
    else:
        # Special for sumProduct
        for k in range(i):
            for m in range(j):
                ndarray[k, m] = torch.from_numpy(ndarray[k, m].astype(np.int64)).to(torch.long).to(device)


# if __name__ == "__main__":
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     # Pre-load HEX Graph on the running device
#     hexG = scipy.io.loadmat("F:\\FGHI\\HEX\\hexG_CUB_genus_class.mat")['hexG']
#     # NxM Cell in Mat = NxM np.array(dtype=object)
#     cliques = hexG['cliques'][0, 0]
#     convert_to_torch(cliques, device)
#     stateSpace = hexG['stateSpace'][0, 0]
#     convert_to_torch(stateSpace, device)
#     variables = hexG['variables'][0, 0]
#     convert_to_torch(variables, device)
#     childVariables = hexG['childVariables'][0, 0]
#     convert_to_torch(childVariables, device)
#     sumProduct = hexG['sumProduct'][0, 0]
#     for i in range(np.size(sumProduct, 0)):
#         convert_to_torch(sumProduct[i, 0], device)
#     varTable = hexG['varTable'][0, 0]
#     for i in range(np.size(varTable, 0)):
#         convert_to_torch(varTable[i, 0], device)
#     upMsgTable = hexG['upMsgTable'][0, 0]
#     convert_to_torch(upMsgTable, device)
#     downMsgTable = hexG['downMsgTable'][0, 0]
#     convert_to_torch(downMsgTable, device)
#     numVar = hexG['numVar'][0, 0]
#     numVar = torch.from_numpy(numVar.astype(np.int64)).to(torch.long).to(device)
#     cliqParents = hexG['cliqParents'][0, 0]
#     cliqParents = torch.from_numpy(cliqParents.astype(np.int64)).to(torch.long).to(device)
#     upPass = hexG['upPass'][0, 0]
#     upPass = torch.from_numpy(upPass.astype(np.int64)).to(torch.long).to(device)
#
#     Eh = scipy.io.loadmat("F:\\FGHI\\HEX\\Subsumption_CUB_genus_class.mat")['Eh']
#     Eh = torch.from_numpy(Eh).to(torch.long).to(device)
#
#     # fs = scipy.io.loadmat("F:\\FGHI\\HEX\\hex_test2_f.mat")['f']
#     # fs = torch.from_numpy(fs).requires_grad_(True).to(device)
#     # labels = scipy.io.loadmat("F:\\FGHI\\HEX\\hex_test2_y.mat")['y'].astype(np.int64)
#     # labels = torch.from_numpy(labels).to(torch.long).to(device).squeeze(0)
#     fs = torch.ones(64, 322).to(torch.float64).to(device) * 0.5
#     labels = torch.ones(64).to(torch.long).to(device) * 15
#     labels = labels - 1
#
#     criterion = HEXLoss(cliques, stateSpace, numVar, cliqParents, childVariables, upPass,
#                         sumProduct, upMsgTable, downMsgTable, variables, varTable, Eh)
#     loss = criterion(fs, labels, device)
#     print("finished")

#     p1 = pMargin..cpu().detach().numpy()
#     z1 = z.cpu().detach().numpy()
#     scipy.io.savemat("F:\\FGHI\\HEX\\hexG_res3.mat", {'p1': p1, 'z1': z1})