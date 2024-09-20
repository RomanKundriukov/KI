using KI_TensorFlow_Keras.CNN;
using KI_TensorFlow_Keras.Services;

namespace KI_TensorFlow_Keras
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            string originalDir = Path.Combine("C://Roman/ML/train", "train");
            string newBaseDir = Path.Combine("C://Roman/ML", "cats_vs_dogs_small");

            DogVSCat d = new DogVSCat();
            //var trainData = d.createTrainData("C://Roman/ML");

            if (true)
            {
                var model = d.createModel();

                string trainPath = Path.Combine(newBaseDir, "train");
                var trainDataset = ImageLoader.LoadImagesFromDirectory(directoryPath: trainPath, 180, 180, 32);
            }

        }

    }
}
