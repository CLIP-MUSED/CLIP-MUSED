edim=512
depth=18
data_root='/data/home/zhouqy/multi_subject/dataset/HCP'


CUDA_VISIBLE_DEVICES=7 python3 -W ignore decoding_HCP_MS.py -m='train' -e=1 -edim=512 -depth=18 -usecls -hfmodel='aver_cap_concat_vitb16_clip' -lfmodel='img_vitb16_clip' -hdim_rd='not_rd' -ldim_rd='not_rd' -hlayer='-1' -llayer='0' -otype='F' -coe_clf=1 -coe_orth=1e-3 -coe_rdm_hlv=1e-3 -coe_rdm_llv=1e-1 -fea_pp -thres=1.5 -sel_label -abkind='label_0p1'

for sub in '233326' '172130' '951457' '191336' '169040' '878776' '169343' '102816' '573249'
do
CUDA_VISIBLE_DEVICES=7 python3 -W ignore decoding_HCP_MS.py -m='test' -edim=512 -depth=18 -usecls -sel_label --subject=$sub -me=-1 -abkind='label_0p1'
done