# Global maximum and minimum values obtained empirically and used to normalize SAR images
M = 10.089038980848645
m = -1.429329123112601

# Must be a power of 2 lower than min(height, width) of the image to despeckle
# Default to 256 as the trained spotlight and stripmap models stored in merlin/saved_model
PATCH_SIZE = 256

# Has to be lower than the PATCH_SIZE
# Default to PATCH_SIZE - 2 in order to not have visible borders in the despeckled images
STRIDE_SIZE = PATCH_SIZE - 2
