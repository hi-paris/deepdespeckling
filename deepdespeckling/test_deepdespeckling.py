from deepdespeckling.despeckling import despeckle, despeckle_from_coordinates, despeckle_from_crop

image_path = "/Users/hadrienmariaccia/Documents/Projects/deepdespeckling/img/entire/sar2sar_tests"
destination_directory = "/Users/hadrienmariaccia/Documents/Projects/deepdespeckling/img/entire/sar2sar_tests"
coordinates_dictionnary = {'x_start': 0,
                           'y_start': 0, 'x_end': 700, 'y_end': 700}

despeckle_from_crop(image_path, destination_directory,
                    model_name="sar2sar")
