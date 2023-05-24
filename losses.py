import torch
def cal_rdm(mat1, mat2):
    # calculate RDM between samples
    sample_num = mat1.shape[0]
    rdms = torch.zeros((sample_num, sample_num)).cuda()
    num = torch.mm(mat1, torch.t(mat2))
    mat1_norm = torch.norm(mat1, p=2, dim=1, keepdim=True)
    mat2_norm = torch.norm(mat2, p=2, dim=1, keepdim=True)
    den = torch.mm(mat1_norm, torch.t(mat2_norm))
    rdms = 1. - torch.mul(num, 1/(den+1e-6))    # in case of nan

    return rdms

def calculate_orthogonal_regularization_L1(feature1, feature2):
    loss = 0.0
    loss = feature1 * feature2
    loss = loss.sum(1)
    loss = torch.abs(loss)
    loss = loss.sum()
    loss /= feature1.shape[0]
    return loss

def calculate_orthogonal_regularization_L2(feature1, feature2):
    loss = 0.0
    loss = feature1 * feature2
    loss = loss.sum(1)
    loss = loss ** 2
    loss = loss.sum()
    loss /= feature1.shape[0]
    return loss

def calculate_orthogonal_regularization_F(feature1, feature2):
    loss = 0.0
    feature2_T = feature2.t()
    loss = feature1.mm(feature2_T)
    loss = loss ** 2
    loss = loss.sum()
    loss /= (feature1.shape[0] * feature2.shape[0])
    return loss

def regularization_F(mat):
    loss = mat ** 2
    loss = loss.sum()
    loss /= (mat.shape[0] * mat.shape[1])
    return loss
