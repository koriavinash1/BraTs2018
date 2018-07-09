import SimpleITK as sitk
import numpy as np
from scipy.ndimage.measurements import label
from skimage.morphology import erosion, dilation
import nibabel as nib
import os
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
from pydensecrf.utils import compute_unary, create_pairwise_bilateral, create_pairwise_gaussian, softmax_to_unary


normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transformList = []
transformList.append(transforms.ToTensor())
transformList.append(normalize)
transformSequence=transforms.Compose(transformList)

class postProcessing(object):
	"""docstring for postProcessing"""
	def __init__(self):
		super(postProcessing, self).__init__()

	def dilation(self, vol):

		return dilated_vol

	def erode(self, vol):

		return eroded_vol

	def readVol(self, path):
		data = nib.load(path)
		self.affine = data.get_affine()
		return data.get_data()

	def saveVol(self, vol, path, verbose = False):
		img = nib.Nifti1Image(vol, self.affine)
		img.set_data_dtype(np.uint8)
		nib.save(img, path)
		if verbose: print ("Volume saved in path : {}".format(path))
		pass

	def connectedComp(self, vol, thresh = 0.50, verbose=False):
		"""
			helps in removing false positives predicted based
			on volumes of each segmentation maps obtained

			input args:
				>> vol 3D volume data (segmentation map)
				>> thresh threshold value to remove volumes
				>> verbose to print max and min for each volume

			output args:
				>> 3D volume with same shape as input volume
		"""

		label_vol, label_masks = label(vol)
		volumes = []
		for i in range(label_masks):
			x, y, z = np.where(label_vol == i+1)
			v = (np.max(x) - np.min(x)) * (np.max(y) - np.min(y)) * (np.max(z) - np.min(z))
			volumes.append(v)

		if verbose:	print ("Max. Volume: {}".format(np.max(volumes)) + " Min. Volumes: {}".format(np.min(volumes)))

		for i in range(len(volumes)):
			if volumes[i] < np.max(volumes) * thresh:
				label_vol[label_vol == i+1] = 0

		label_vol[label_vol > 0] = 1
		return label_vol * vol

	def crf(self, input_vol, softmax_vol):
		# prob shape: classes, width, height, depth
		# vol shape: width, height, depth

		assert len(softmax_vol.shape) == 4
		assert len(input_vol.shape) == 4
		return_ = np.empty((input_vol.shape[1:]))
		for slice_idx in range(input_vol.shape[3]):
			image   = input_vol[:, :, :, slice_idx]
			softmax = softmax_vol[:, :, :, slice_idx]

			# softmax = softmax.transpose(2, 0, 1)
			# image   = image.transpose(2, 0, 1)

			n_classes = softmax.shape[0]
			unary = unary_from_softmax(softmax)
			d = dcrf.DenseCRF(image.shape[1]*image.shape[0], n_classes)
			d.setUnaryEnergy(unary)
			feats = create_pairwise_gaussian(sdims=(1., 1.), shape=image.shape[:2])
			d.addPairwiseEnergy(feats, compat=3,
			                    kernel=dcrf.DIAG_KERNEL,
			                    normalization=dcrf.NORMALIZE_SYMMETRIC)


			# This creates the color-dependent features and then add them to the CRF
			feats = create_pairwise_bilateral(sdims=(3., 3.), schan=0.5,
			                                  img=image, chdim=2)
			d.addPairwiseEnergy(feats, compat=10,
			                    kernel=dcrf.DIAG_KERNEL,
			                    normalization=dcrf.NORMALIZE_SYMMETRIC)
			Q = d.inference(iterations)
			probabilities = np.array(Q).reshape((n_classes,image.shape[0],-1))
			labels = np.argmax(Q, axis=0).reshape(image.shape[0:2])
			return_[:,:,slice_idx] = labels
		return return_


pp = postProcessing()
if __name__ == "__main__":
	root_path = '/media/brats/MyPassport/Avinash/Kaminstas_2018/MICCAI_BraTS_2018_Data_Training/test'
	files = next(os.walk(root_path))[1]

	for _file in files:
		sequences = os.listdir(volume_path+'/'+p)
        for s in sequences:
            if 'flair' in s:
                flair_volume= nib.load(volume_path+'/'+p+'/'+s)
            if 't2' in s:
                t2_volume= nib.load(volume_path+'/'+p+'/'+s)
            if 't1ce' in s:
                t1ce_volume= nib.load(volume_path+'/'+p+'/'+s)
        affine = flair_volume.get_affine()
        flair_volume= flair_volume.get_data()
        t2_volume   = t2_volume.get_data()
        t1ce_volume = t1ce_volume.get_data()

        input_volume = np.empty((3, flair_volume.shape[0],flair_volume.shape[1],flair_volume.shape[2]))
        input_volume[0,:,:,:] = flair_volume
        input_volume[1,:,:,:] = t2_volume
        input_volume[2,:,:,:] = t1ce_volume

		path = os.path.join(os.path.join(root_path, _file), 'get_2D_tiramisu_softmax.nii.gz')
		print (path)
		vol = pp.readVol(path)
		vol = pp.crf(input_volume, vol)
		# print (vol.shape)
		vol = pp.connectedComp(vol, 0.50)
		

		vol[vol == 3] = 4
		save_path = path.replace('get_2D_tiramisu_softmax', '2D_tiramisu_crf')
		pp.saveVol(vol, save_path)
