from deepdespeckling.merlin.inference.despeckling import despeckle_from_coordinates

image_path = "/Users/hadrienmariaccia/Documents/Projects/deepdespeckling/img/entire"
destination_directory = "/Users/hadrienmariaccia/Documents/Projects/deepdespeckling/img/entire"
model_weights_path = "/Users/hadrienmariaccia/Documents/Projects/deepdespeckling/deepdespeckling/merlin/inference/saved_model/spotlight.pth"
coordinates_dictionnary = {'x_start': 0,
                           'y_start': 0, 'x_end': 700, 'y_end': 700}

despeckle_from_coordinates(image_path, coordinates_dictionnary, destination_directory,
                           model_weights_path=model_weights_path)
