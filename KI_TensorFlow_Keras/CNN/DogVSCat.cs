using static Tensorflow.KerasApi;

namespace KI_TensorFlow_Keras.CNN
{
    public class DogVSCat
    {
        public bool createTrainData(string path)
        {
            if (Directory.Exists(path))
            {
                string startPath = Path.Combine(path, "train");
                string newBaseDir = Path.Combine(path, "cats_vs_dogs_small");
                string originalDir = Path.Combine(startPath, "train");
                makeSubset(subsetName: "train", startIndex: 0, endIndex: 1000, originalDir: originalDir, newBaseDir: newBaseDir);
                makeSubset(subsetName: "validation", startIndex: 1000, endIndex: 1500, originalDir: originalDir, newBaseDir: newBaseDir);
                makeSubset(subsetName: "test", startIndex: 1500, endIndex: 2500, originalDir: originalDir, newBaseDir: newBaseDir);

            }

            return true;
        }

        public void makeSubset(string subsetName, int startIndex, int endIndex, string originalDir, string newBaseDir)
        {
            string[] categories = { "cat", "dog" };

            foreach (var category in categories)
            {
                string categoryDir = Path.Combine(newBaseDir, subsetName, category);
                Directory.CreateDirectory(categoryDir);

                for (int i = startIndex; i < endIndex; i++)
                {
                    string fileName = $"{category}.{i}.jpg";
                    string sourceFile = Path.Combine(originalDir, fileName);
                    string destFile = Path.Combine(categoryDir, fileName);

                    if (File.Exists(sourceFile))
                    {
                        File.Copy(sourceFile, destFile, overwrite: true);
                    }
                    else
                    {
                        Console.WriteLine($"File {fileName} wurde nicht in {originalDir} gefunden");
                    }
                }
            }
        }

        public object createModel()
        {
            var layers = keras.layers;
            var inputs = keras.Input(shape: (180, 180, 3));

            var x = layers.Rescaling(scale: 1.0f / 255).Apply(inputs);
            x = layers.Conv2D(filters: 32, kernel_size: (3, 3), activation: "relu").Apply(x);
            x = layers.MaxPooling2D(pool_size: (2, 2)).Apply(x);
            x = layers.Conv2D(filters: 64, kernel_size: (3, 3), activation: "relu").Apply(x);
            x = layers.MaxPooling2D(pool_size: (2, 2)).Apply(x);
            x = layers.Conv2D(filters: 128, kernel_size: (3, 3), activation: "relu").Apply(x);
            x = layers.MaxPooling2D(pool_size: (2, 2)).Apply(x);
            x = layers.Conv2D(filters: 256, kernel_size: (3, 3), activation: "relu").Apply(x);
            x = layers.MaxPooling2D(pool_size: (2, 2)).Apply(x);
            x = layers.Conv2D(filters: 256, kernel_size: (3, 3), activation: "relu").Apply(x);
            x = layers.Flatten().Apply(x);

            var outputs = layers.Dense(units: 1, activation: "sigmoid").Apply(x);

            var model = keras.Model(inputs, outputs);

            model.compile(optimizer: "rmsprop", loss: "binary_crossentropy", metrics: new string[] { "accuracy" });
            model.summary();

            return model;
        }
    }
}
