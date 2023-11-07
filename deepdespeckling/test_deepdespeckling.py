from deepdespeckling.merlin.inference.despeckling import despeckle

image_path = "/Users/hadrienmariaccia/Documents/Projects/deepdespeckling/img/entire"
destination_directory = "/Users/hadrienmariaccia/Documents/Projects/deepdespeckling/img/entire"
model_weights_path = "/Users/hadrienmariaccia/Documents/Projects/deepdespeckling/deepdespeckling/merlin/inference/saved_model/spotlight.pth"
coordinates_dictionnary = {'x_start': 0,
                           'y_start': 0, 'x_end': 1200, 'y_end': 1200}

despeckle(image_path, destination_directory,
          model_weights_path=model_weights_path)
