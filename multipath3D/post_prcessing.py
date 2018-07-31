import SimpleITK as sitk
import numpy as np
from scipy.ndimage.measurements import label
from skimage.morphology import erosion, dilation
import nibabel as nib
import os


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
		if verbose: print "Volume saved in path : {}".format(path)
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

		if verbose:	print "Max. Volume: {}".format(np.max(volumes)) + " Min. Volumes: {}".format(np.min(volumes))

		for i in range(len(volumes)):
			if volumes[i] < np.max(volumes) * thresh:
				label_vol[label_vol == i+1] = 0

		label_vol[label_vol > 0] = 1
		return label_vol * vol


pp = postProcessing()
if __name__ == "__main__":
	root_path = './gen_test'
	files = next(os.walk(root_path))[1]

	for _file in files:
		path = os.path.join(os.path.join(root_path, _file), '/get_MRCNN_prediction_resnet.nii.gz')
		vol = pp.readVol(path)
		new = pp.connectedComp(vol, 0.50)

		save_path = path.replace('get_MRCNN_prediction_resnet', 'postProcessed')
		pp.saveVol(new, save_path)
