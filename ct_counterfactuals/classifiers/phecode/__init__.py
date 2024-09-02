import argparse
import numpy as np
import pandas as pd
import sys
import os
import torch
import pathlib
import torch.nn as nn
from . import classes
import torchvision
import latentshift
from . import i3res
import copy


thisfolder = os.path.dirname(__file__)

class PheCodeClassifier(nn.Module):

    def __init__(self):
        super(PheCodeClassifier, self).__init__()

        self.resolution = 224
        self.crop_size = 224
        
        _MODEL_PATH = (
            pathlib.Path(latentshift.utils.get_cache_folder()) / "i3_resnet_best_clip_04-02-2024_23-21-36_epoch_99.pt"
        )


        resnet = torchvision.models.resnet152(pretrained=True)
        self.model = i3res.I3ResNet(copy.deepcopy(resnet), class_nb=1692, conv_class=True)
        
        self.model.load_state_dict(torch.load(_MODEL_PATH, map_location=torch.device('cpu')))
        self.model = self.model.eval()
        
    
        self.upsample = torch.nn.Upsample(size=(self.resolution, self.resolution), mode='bilinear', align_corners=False)
        self.crop = torchvision.transforms.CenterCrop(self.crop_size)
        
        self.csv = _map_to_phenotypes()
        self.targets = self.csv.phecode_str.tolist()

    def forward(self, x):
        x = self.upsample(x[0].swapaxes(1,3)).swapaxes(1,3)[None,...]
        
        assert x.shape[-3:-1] == (self.crop_size,self.crop_size)
        preds = self.model(x)[1]
        preds = torch.sigmoid(preds)
        
        return preds
    

def _map_to_phenotypes():

    _ICD10_PHECODE_PATH = (
        pathlib.Path(thisfolder) / "data" / "Phecode_map_v1_2_icd10cm_beta.csv"
    )
        
    _ICD9_PHECODE_PATH = pathlib.Path(thisfolder) / "data" / "phecode_icd9_rolled.csv"

    
    class_floats = classes.DENSENET_121_PHECODES_EXPANDED
    class_strings = [str(x) for x in class_floats]

    icd10_phecodes = pd.read_csv(_ICD10_PHECODE_PATH)
    icd10_phecodes = icd10_phecodes[["phecode", "phecode_str"]]
    icd10_phecodes.loc[:, "phecode"] = icd10_phecodes.loc[:, "phecode"].astype(str)
    icd10_phecodes.loc[:, "phecode"] = icd10_phecodes.loc[:, "phecode"].apply(
        lambda x: x.rstrip("0").rstrip(".") if "." in x else x
    )
    icd10_phecodes = icd10_phecodes.drop_duplicates(subset=["phecode"])

    # do same for icd9_phecodes
    icd9_phecodes = pd.read_csv(_ICD9_PHECODE_PATH)
    icd9_phecodes = icd9_phecodes[["PheCode", "Phenotype"]]
    icd9_phecodes.columns = ["phecode", "phecode_str"]
    icd9_phecodes.loc[:, "phecode"] = icd9_phecodes.loc[:, "phecode"].astype(str)
    icd9_phecodes.loc[:, "phecode"] = icd9_phecodes.loc[:, "phecode"].apply(
        lambda x: x.rstrip("0").rstrip(".") if "." in x else x
    )
    icd9_phecodes = icd9_phecodes.drop_duplicates(subset=["phecode"])
    icd_phecodes = pd.concat([icd10_phecodes, icd9_phecodes])
    icd_phecodes = icd_phecodes.drop_duplicates(subset=["phecode"])

    class_strings_df = pd.DataFrame(class_strings, columns=["phecode"])
    class_strings_df = class_strings_df.merge(icd_phecodes, how="inner", on="phecode")
    #class_strings_df.loc[:, "prediction"] = predictions[0, :]
    return class_strings_df
    #print("Predictions:")
    #print(class_strings_df.to_string(index=False))

thresholds = [-3.9142807, -3.8201687, -8.335198, -4.0526714, -5.524053, 1000, -5.601118, -9.093004, -2.6247005, -4.8611517, -5.692639, -4.4767666, -2.7452593, -4.7756944, -5.15854, -5.4859376, -3.7111378, -10.278203, -4.6990175, -5.930388, -5.850779, -5.8267074, -10.063895, -4.5129, -3.553463, -7.3598766, -5.2601647, -4.1694927, -4.698262, -5.2981305, -5.282896, -5.2748437, -8.445142, -4.8848505, -12.453269, -4.5291157, -4.687997, -3.5341504, -4.356753, -6.7389216, -7.368476, -6.716555, -6.1895704, -11.368503, 1000, -6.164184, -7.987862, -6.884839, -7.0462317, -8.103411, -3.5390375, -9.270291, -5.0003424, -10.250701, -12.992682, 1000, -6.5584073, -6.024396, -9.485866, -7.0039043, -6.416015, 1000, -11.82238, -8.678655, -9.01976, -4.3565445, -4.8336797, -6.4669666, 1000, -12.683268, 1000, -4.364211, -7.073495, -9.505942, 1000, -6.161602, -6.2472873, -7.709076, -6.23533, -4.363092, -3.28676, -2.9924815, -4.298174, -4.1091704, -5.1369023, -2.8517823, -4.954764, -3.8513327, -5.631405, -3.6552043, -5.0939145, -7.200494, -4.6000056, -4.722744, -6.441061, -4.87743, -4.032107, -6.594639, -5.5922976, -6.143703, -6.9329453, -7.2785077, -8.746354, -4.0131536, -5.5621266, -4.4844418, 1000, 1000, -5.140557, -5.4038877, -8.537478, -4.4599237, -3.8256874, -4.974232, -5.5584526, -5.859809, -3.4238994, -5.458725, -6.2133913, -5.4409966, -3.670884, -4.7638144, -5.361941, -8.881918, -3.9792418, -4.2707624, -5.425354, -9.813952, -6.7469416, -6.8302627, -6.723822, -5.797017, -5.3207917, -2.8396754, -3.6065629, -9.282811, -6.551751, -2.7427928, -2.562125, -3.6893978, -4.0946035, -3.9480028, -4.539352, -6.204186, -5.5511317, -8.106589, -4.4203525, -9.002151, -4.3079386, -8.087354, -5.0610223, -4.244021, -4.3926344, -7.1387873, -4.739801, -5.609912, -6.44809, -4.156305, -4.75837, -5.282153, -6.38088, -5.3194995, -4.759424, -7.457852, 1000, -5.801065, -3.8924434, -5.72378, 1000, -5.261749, -9.047877, -7.208177, -5.605255, 1000, -6.484562, -9.487115, 1000, -11.391946, -11.085152, -4.3117704, -9.483497, -6.4540987, -19.688469, -8.51596, 1000, -6.3907175, -7.1414485, 1000, -5.1334476, 1000, -7.1297793, -4.568911, 1000, -7.555632, 1000, 1000, -8.892485, -5.3892946, -5.193599, -8.156236, -5.3146873, -6.8202667, -7.917229, 1000, -5.838333, -6.3368917, -2.5868914, -16.29499, -6.2018967, -5.8867464, -6.454059, 1000, -10.827515, -6.073224, -2.6975205, -5.3262043, -6.2649364, -11.879592, -10.079556, -7.8858047, 1000, -2.4758048, -5.954743, -5.2915792, -6.7808385, -5.524318, -7.852188, -4.285049, -6.9020333, -3.4699004, -8.290389, -7.3007293, -6.804897, -4.850072, -4.8539, -4.5431676, -5.0800624, -5.099609, -8.924243, -5.275433, -8.690647, -10.329384, -7.267658, -8.201425, 1000, -5.6173744, -19.170307, -4.304783, -6.4546595, -6.4210005, 1000, -5.1216903, -5.1385, 1000, -6.6324997, -6.8011813, -7.376037, -6.002939, -6.486258, -6.5066595, -8.218658, 1000, -2.6215842, -5.841871, -3.3100357, -6.974983, -5.38717, -4.953932, -3.9590778, -6.852117, -4.6498556, -10.34998, -5.557645, -18.543415, -5.13892, 1000, -6.8824177, -5.874969, 1000, -5.204841, -8.141942, -6.15151, -6.4107103, -6.5502615, -7.127973, -7.050305, -3.8567686, -9.646249, -5.5937595, -5.4845767, -10.806036, -11.494918, -4.04443, -4.8439918, 1000, -1.6756172, -4.486887, -4.8515615, -6.0042424, -10.975703, -8.01585, -4.4278316, -6.100668, -9.222965, -9.280701, -3.9987662, -6.151721, -9.958501, -7.7286906, -4.627808, -4.790248, -4.166577, -5.6379285, -5.585715, -1.7815578, -4.982662, -2.8564966, -4.905824, -3.192518, -2.9742198, -3.7375863, -5.0475817, -2.9644978, -4.6751814, -9.339716, -4.0021157, 1000, -4.751873, -6.300547, -7.44451, 1000, -8.369592, 1000, -2.5613816, -2.9961734, -4.004129, -6.7724094, -7.8860507, -4.6092043, -5.218015, -6.406462, -3.558882, -6.8391685, -8.523907, -3.9832222, -4.5846434, -5.9850183, -6.7441206, -7.269253, -6.2531223, -16.367968, -5.328518, -7.187426, -5.5138984, -4.1191144, -6.9792504, -8.212743, -6.8885603, -8.629377, -4.275159, -4.5419116, -6.7772427, -2.0124087, -3.0057063, -3.433406, -5.678539, -5.7275515, -6.6165705, -6.3858747, -8.246655, -16.807768, -7.092334, -4.243525, -8.798621, -6.02382, -8.136053, -6.0196853, -4.414514, -5.775307, -5.718316, -4.6350656, -3.4149587, -8.106433, -10.275709, -3.2934868, -6.1409125, -5.5684495, -7.1024847, -1.8090839, -4.119813, -3.9950411, -2.3513892, -6.4354587, -2.937714, -6.8616586, -8.143391, -3.1252987, -4.5237756, -6.9747877, -6.510592, -3.3381004, -4.887235, -6.374405, -6.6720176, -8.9258585, -7.605266, -3.6847272, -6.122732, -9.028674, -5.997324, -4.3758926, -3.8577015, -6.20305, -6.2671046, -14.192004, -5.917912, -6.1742673, -3.5330734, -4.501637, -5.696377, -6.1080728, -6.075609, -4.841079, -5.580992, -7.947659, -6.368203, -2.7708, -4.5287695, -3.0669165, -3.1183329, -6.7969594, -6.8520937, -8.3858385, -2.6539588, -3.106154, -3.4525402, -6.479821, -6.134695, -8.965971, -5.5694027, -8.552359, -6.1848087, -8.670681, -9.290557, -7.1043367, -7.332397, -6.8063483, -5.781487, 1000, -8.908566, -4.279232, -7.6715727, -8.012348, -6.378815, -7.344786, -6.720323, 1000, 1000, -5.2468905, -5.441297, -7.2057586, -6.783953, -4.496314, -9.2351, -7.2362075, -12.559132, -3.5371108, -7.725563, -4.058654, -4.123184, -5.5009646, -1.265666, -5.955208, -8.66473, -11.757166, -9.526501, -6.2795463, -6.2074814, -2.5638182, 1000, -2.9486065, -5.815131, -3.2078705, -3.9692965, -7.8801055, 1000, -6.8013067, -6.1559987, -6.1422415, -5.9269276, -6.777363, -7.709606, -6.270835, -6.288548, -6.818769, -6.202619, -17.738123, -8.555968, 1000, -5.0331316, -6.314584, -6.922768, -7.0717587, -5.2562056, -5.3140616, -8.849476, -2.7424974, -2.908768, -3.1766434, -3.7516909, -7.116419, -8.136805, -5.9700265, -7.2326655, -4.25555, -4.06342, -7.2768874, -8.051524, -8.532431, -5.1197433, -4.954379, -5.476675, -6.0865874, -7.8762727, -10.213842, -4.4193373, -5.9617276, -10.3733225, -6.6448355, -5.0505958, -5.735439, -4.7760406, -5.694214, -5.4430118, -6.0739727, -8.616823, -7.453302, -4.946371, -6.291183, -8.044804, -7.2076244, -6.636903, -10.811859, -9.400423, -6.2722793, -6.223826, -5.29022, -5.0655265, -8.034555, -8.123432, -7.3530645, -6.0205526, -7.1203775, -8.226805, -5.7486835, -9.218084, 1000, -6.462771, -7.696025, -6.7421875, -9.293672, -9.019515, -4.5758786, -6.9269867, -5.9302974, 1000, -8.662471, 1000, -5.3470244, -12.067056, -7.476927, -9.433427, -14.362826, -18.305363, -9.029195, -12.282159, 1000, -8.460233, 1000, -10.583963, 1000, -7.7861204, -4.9512954, -6.5789943, -7.765857, -8.565937, 1000, -4.808057, -10.964686, -6.3792043, -3.6890843, -7.0578523, -7.286211, -7.4501767, -8.759174, -6.5241294, -5.399376, -10.902813, -7.9185305, 1000, -8.00317, -17.661089, 1000, -6.7938313, -7.1889286, -5.650126, -9.132275, -5.4145336, -8.09096, 1000, -8.224857, -8.971927, 1000, -7.2165575, -7.7532825, -7.6364064, -7.6295853, -6.0184946, -8.036484, -12.501551, -8.197742, -9.240424, -7.2891912, -6.49191, -20.334938, 1000, -8.160657, -9.844914, 1000, -5.934516, -13.409002, -5.944948, -6.0269656, -5.0436096, -11.21136, -5.268137, -7.2197757, -9.765955, -11.422366, 1000, -7.136137, -5.6819053, 1000, -6.379094, -7.6456733, -8.178648, -10.062997, -9.965457, 1000, 1000, 1000, 1000, 1000, -10.840116, 1000, 1000, -3.4496784, 1000, -5.572351, -18.454292, -3.4817624, 1000, -3.6595976, -4.50277, -10.125995, 1000, -8.525665, -4.9043465, -7.2773504, -5.310391, 1000, -7.2553086, -6.9380913, -3.6259027, -5.438766, -5.0544624, -6.1034927, -9.266862, -5.342852, -4.8316674, -1.459188, -1.2433839, -4.331108, -5.569959, -4.364206, -5.9946747, -6.370476, -7.8311744, -3.501435, -6.945404, -3.1319492, -12.768268, -4.7186255, -5.357293, -6.296709, -10.462001, -3.4000676, -4.431441, -4.4047375, -4.561304, -5.37582, -2.7872894, -2.5357568, -5.9482493, -7.1301923, -4.755447, -7.876433, -7.2722507, -6.9871593, -4.734725, -6.176027, -8.032518, -7.1443887, -7.817216, -1.605716, -4.5578327, -4.375106, 1000, -7.4068394, -6.418318, -8.9599905, -3.7095778, -4.5723205, -4.4779553, -6.4800124, -1.532854, -3.8052435, -4.2915864, -3.88702, -5.4490986, -4.4018435, -5.0083632, -4.970474, -2.8475463, -4.7256556, -2.518789, -8.528262, -5.600958, -5.1550937, -3.5188038, -4.3312845, -1.9956285, -5.832868, -5.641454, -3.6498897, -7.267234, -4.608581, -4.262268, -5.454524, -5.7569933, -4.6994777, -7.561002, -6.2215967, -6.67942, -8.021656, -7.246777, -4.127468, -4.6076884, -5.941004, -7.045001, -4.1635804, -4.1727448, -4.2508163, -4.161892, -4.382606, -5.9117613, -7.451072, -5.1843386, -4.4986153, -6.674818, -7.0438004, -9.946617, -8.430754, -5.0800753, -4.117022, -3.7045326, -5.6269965, -4.800261, -5.7518735, -5.6677117, -6.39918, -8.106369, -6.0236897, -5.5618935, -6.6230545, -9.863524, -8.866798, -4.9028506, -4.618087, -5.531319, -7.8970613, -9.825365, -10.123688, -7.342958, -7.036355, -7.1429343, -5.858937, 1000, -7.445101, -8.360444, -5.5655074, -7.0541363, -9.149191, -13.068719, -5.6058655, -6.421198, -7.046583, -2.779703, -4.2945633, -4.3937545, -18.827816, -5.289244, -7.158425, -8.922014, -4.572553, -6.2580137, -6.0395327, -3.9038796, -3.005371, -5.651299, -5.2758603, -3.1272428, -4.198854, -6.106627, -5.4459105, -5.3762264, -7.8386354, -5.0525775, -5.752947, -8.493562, -8.223254, 1000, -7.7592907, -5.117841, 1000, -6.3299937, -5.9288316, -8.129693, 1000, -8.78261, -5.2763276, -7.186724, -4.4858828, -6.4169455, -6.419972, -4.7440205, -3.6230712, -5.943242, -6.527569, -7.3970466, -8.890329, -7.947094, -6.3986516, -6.035809, -6.555307, -5.2569075, -2.7522354, -5.962473, 1000, -6.8420105, -3.083238, -5.5376673, -5.569244, -5.889883, -5.76198, -6.9037633, -8.870113, -5.997158, -5.3641787, -5.7520437, -4.664925, -5.474176, -4.169892, -5.0254817, -9.077085, -5.913703, -4.233134, -2.304065, -2.6939766, -4.7442565, -5.6624002, -4.7917757, 1000, -3.915492, -4.0254393, -6.415914, -6.0448093, -5.4067564, -7.0522566, -2.5310318, -3.5274923, -4.0890255, -4.208872, -6.4662375, 1000, -7.199396, -5.1506615, -5.6408052, -1.6397461, -11.109257, -3.4218853, -6.1618366, -5.9534893, -3.5695562, -8.283925, -3.588137, -6.318225, -4.2420454, 1000, 1000, 1000, -8.092687, -8.050409, 1000, -6.676379, 1000, -6.7155433, -8.315038, 1000, 1000, 1000, -7.1907897, -8.879716, -7.924751, -9.705819, -7.101748, 1000, 1000, 1000, 1000, 1000, -8.304142, -12.009675, -6.7088547, 1000, -7.913427, -6.8797994, -11.153666, -4.856571, -5.212509, -6.0075865, -8.916868, 1000, -7.725044, -11.065315, -7.059444, -8.629013, 1000, -2.4347847, -2.7747035, -2.7886248, -6.735376, -5.9574413, -6.5166883, -11.675734, -4.1753526, -7.345109, -5.9296184, -5.7674627, -6.590238, -6.9231076, -5.386577, -5.317944, -4.987712, -3.934799, -6.989897, -3.8228686, -5.731439, -5.472957, -6.1923327, -6.176691, -4.8355174, -4.0471597, -5.2524304, -4.7124395, -4.9440575, -3.5620136, -5.460271, -3.6831894, -4.2821484, -5.0186577, -5.2553473, -3.2876637, -5.923059, -3.862584, -10.435966, -4.0325303, -4.10136, -5.1740746, -3.1443326, -3.8084254, -5.1478877, -7.0204453, -6.0782986, -9.3863735, -6.190032, -6.7328835, -3.2862768, -3.9537327, -3.0220578, -5.818366, -4.8651547, -2.0547688, -1.30279, -2.0588326, -2.2550702, -2.8460298, -3.4359915, -1.9647441, -2.0373554, -4.4349127, -2.6549702, -4.968556, -3.7666872, -5.1812053, -2.927736, -3.1261744, -3.7957697, -2.0684628, -6.1467733, -3.985593, -2.41073, -4.3611755, -6.799636, -2.9157312, -4.0956435, -1.9890534, -1.9864831, -6.9636674, -6.0294967, -4.5478005, -3.6479897, -4.4931607, -4.6846833, -5.3583426, -4.4472837, -3.1909504, -4.941123, -5.234858, -4.911931, -3.7193606, -4.2743063, -4.1619563, -5.3773894, -3.4944296, -3.0079708, -6.90931, -3.0557761, -3.1551144, -4.474759, -3.7702253, -5.978424, -3.58503, -4.43322, -3.6665223, -2.5197268, -3.8642333, -4.453978, -9.796065, -8.9515505, 1000, 1000, -9.010094, -6.137004, -5.855067, -5.910466, -4.8905067, -3.0784523, -6.645617, -4.0671353, -5.8794403, -6.121621, -4.264016, -6.8373485, -6.258978, -2.570778, -16.927624, 1000, -4.9776044, -6.7176228, -4.148611, -5.1078053, -6.601779, -7.150152, -7.638493, -7.6967883, -4.1063643, -2.8328373, -4.43007, -4.506104, -6.848836, -6.373291, -10.798324, -3.3736598, -5.500399, -6.278487, -2.7875772, -3.2043302, -6.8735757, -5.016022, -4.9304767, -3.1708934, -4.238796, -5.864868, -6.029306, -5.9932656, -6.9766226, -6.9270086, 1000, -4.495361, 1000, -4.5923233, -2.4985316, -6.268746, -4.149538, -3.8304489, -4.52722, -4.9753227, -6.179276, 1000, -6.6725516, -4.652198, -4.159532, -6.427597, -7.7570095, -7.8881392, -5.88738, 1000, -7.784795, -8.266624, -7.547113, -7.433929, -9.279277, -7.142502, 1000, -16.845297, -6.476089, -4.6295295, 1000, -24.558868, -7.410839, 1000, 1000, 1000, -8.135538, -9.039762, 1000, -7.5843544, 1000, -9.392554, -10.047483, 1000, 1000, 1000, 1000, -5.2756095, -6.6189537, -5.1026897, -6.9912486, -6.3707323, -6.4400206, -7.9079347, -6.187892, -10.264123, -6.957906, -7.07731, -9.224945, -5.09806, -4.2035317, -7.6388006, -8.243809, -8.698024, -9.322955, -4.293237, -5.5517197, -4.99845, -7.803121, -5.471726, -7.756577, 1000, 1000, -6.820531, -6.378313, -6.6765532, 1000, -5.38965, -5.6110225, -9.600215, -6.082467, -3.8935523, 1000, -5.315461, -5.861892, -7.7625184, -6.5230265, -7.873914, -7.076467, -10.290618, -8.430091, 1000, -7.640424, -5.643403, -7.5368752, -6.3923955, -8.570926, -10.13319, -7.6039715, -8.306093, -12.385479, -3.53362, -6.955378, 1000, -6.4087505, -16.70004, -4.420626, 1000, -8.769145, 1000, 1000, -8.138276, -9.749295, -5.998365, -21.699963, -4.6469526, 1000, -8.222256, -7.0857444, 1000, -8.320032, -7.0026603, -7.3784924, 1000, -16.786465, 1000, 1000, 1000, -8.971718, -6.0708284, -6.862376, 1000, 1000, 1000, 1000, -8.331855, -8.087349, -6.1072884, -3.7055013, -8.474627, -6.956398, -6.218378, -4.3929877, -4.5095506, -4.621764, -5.3886366, -8.892287, -16.40548, -11.257563, -7.0804014, -11.201434, -3.760328, -4.690237, -5.898692, 1000, -3.5807223, -5.280458, -6.942343, -7.5921197, -12.357978, -10.613037, -7.700958, -7.1983333, -10.0685625, -7.1857142, -9.632919, -17.921352, -9.759549, -6.7446327, -5.426221, -7.4991736, -5.4528804, -6.829386, -7.8508067, 1000, -4.8007107, 1000, 1000, -5.694039, -6.950051, -6.0748873, -5.711575, 1000, -5.5756536, -10.3463745, -7.416255, -6.174637, -9.298282, 1000, -9.167252, -7.306571, -8.275577, -8.630967, -10.507516, -9.221053, -6.9148374, -6.055005, -7.8254695, -20.083055, 1000, -6.8573375, -4.895584, -10.912818, -9.209539, -4.8448215, -10.188798, -7.9217887, -8.460196, -3.1862938, -6.099617, -5.599871, -6.9727864, -7.70324, 1000, -7.9578137, -6.640641, -3.9508584, -6.3004003, -6.838401, -6.252302, -5.711983, 1000, -8.854606, 1000, -7.6130314, -6.8715525, -9.34832, -10.993873, -4.844087, -4.7498913, -8.004439, -5.369234, -6.8373637, -6.2433934, -17.624472, -5.882883, -10.768446, -6.144278, -5.0331116, -5.1302166, -5.8945613, -3.4892712, -3.6317096, -6.75304, -8.250961, -4.578592, -9.489041, -4.893691, 1000, 1000, -7.363998, -9.555189, -10.345595, 1000, 1000, -7.287769, -7.2555585, -5.8782997, -5.706698, 1000, 1000, -7.2823186, -5.8714857, -7.618842, 1000, 1000, -8.095935, -8.191396, -16.398129, -6.1317444, -9.165269, 1000, -8.311255, -10.279444, -3.93762, 1000, -7.6755147, -5.34369, -8.372251, -11.142829, -7.198148, -4.5166774, 1000, -6.1941013, 1000, -9.376963, 1000, -6.8785777, 1000, -5.1042895, -5.8484874, -8.601193, -8.136631, -8.872567, 1000, 1000, 1000, 1000, -18.776773, -4.5014753, -8.863063, -8.91576, -4.9834776, -6.2190347, -6.472572, -6.0605755, -4.0864086, -5.2567167, -4.6115437, -6.8939323, -8.029152, -19.900936, -4.1264267, -5.693023, 1000, -8.332603, -7.151342, -6.880814, 1000, -5.166325, -18.236216, -9.808698, -20.383715, -5.589283, -3.628641, -19.075758, 1000, -5.3421736, -6.745078, -12.554182, 1000, -4.6238, -3.4737263, -5.906809, -6.38873, -8.099891, -6.152294, -5.823035, -7.3161883, -6.610015, -8.390113, 1000, 1000, -6.2435627, -8.868454, -10.45979, 1000, -11.349093, -10.259462, -5.9768457, -6.1910458, -6.1522985, -4.5639877, 1000, -7.4959354, -5.5853024, -6.819849, -5.099764, 1000, -5.135858, -2.5805798, -2.5334928, -6.041728, 1000, 1000, 1000, -6.8712544, -10.054554, -7.2796216, 1000, -7.3127923, -7.5269566, -5.1303535, -6.0410023, 1000, 1000, -7.0857053, -6.050042, 1000, -7.6553698, -8.81224, -5.587692, 1000, -2.6240191, -3.663506, -5.7244463, -6.0968914, -8.703872, -6.575709, -10.684562, -4.3314652, -3.916955, -4.1965094, -6.5074596, -4.405285, -7.1346335, -5.9538054, -4.7998214, -6.6703005, -6.3471637, -4.104202, -4.183397, -5.8962426, -9.162982, -12.5592785, -2.5043142, -5.6249948, -2.069932, -5.4812636, 1.1213918, -4.0789876, -0.81819546, -5.56924, -3.3923388, -7.9928303, -3.4790869, -7.9336004, -6.3601913, -7.5505433, -6.7528334, -7.0256376, -4.2100058, -4.8650017, -7.4309993, -8.149364, -8.466829, -7.341432, -9.807026, -6.1901803, -4.9403753, -6.1499057, -2.3760152, -7.3011365, -4.263684, -5.0793467, -5.512972, -5.890106, -7.2152596, -7.956728, -6.4553323, -6.6993613, -6.227491, -5.0931644, -4.9431744, -6.233063, 1000, -6.2619033, -7.747145, -5.251583, -5.3636103, -5.7045407, -5.776118, -5.835819, -6.142769, -6.2467933, -7.5655766, -5.0799866, -7.71821, 1000, -5.4006944, -8.077384, -12.776802, -7.2788715, -6.4210057, -6.200461, -4.5993342, -2.5836341, -5.886666, -4.56422, -8.783659, -6.658182, -4.5222273, -6.405383, -4.944245, -4.6161947, -4.222555, -7.346643, -8.857843, -6.540675, -6.446805, -6.847241, -8.239219, -3.5797896, -5.3406177, -8.362358, 1000, -8.637659, 1000, -5.9199443, -8.416949, -8.338718, -6.4287524, -7.175548, -3.7332618, 1000, 1000, 1000, 1000, -4.6192675, -4.643101, -4.306012, -9.106433, -9.750527, 1000, -15.740151, -4.265307, -7.6429005, -6.833336, -8.980844, -6.11583, -3.2926931, -10.775313, -6.133361, -12.01893, -4.8569236, -6.946491, -6.091843, -2.1461632, -3.1548715, -2.5930037, -7.8673286, -3.6828806, -3.9439766, -4.337725, -6.241506, -10.06751, -4.8681307, -4.9327755, -5.1222034, -5.318516, -5.6618614, -3.0972495, -3.693758, 1000, -12.426768, -9.776034, -3.541384, -9.021397, -9.581536, -6.007825, -9.344038, -6.1998463, -10.752118, -10.471675, 1000, -8.128986, -3.7656639, -3.0898848, -6.5162444, 1000, -12.502672, 1000, 1000, -19.917883, -3.7997208, -4.7290835, -2.4772863, -3.315858, 1000, -5.160256, -3.6896422, -6.857861, -3.5547671, 1000, -4.021182, -5.3806353, -3.3341622, 0.10840053, -7.7254677, -5.8544607, -5.8574095, -5.229862, -3.0508888, -4.1245904, -0.6996384, -3.771353, -8.658906, -3.3887193, -7.5035987, -3.7596714, -4.6752276, -6.1496096, -1.587549, -3.467206]