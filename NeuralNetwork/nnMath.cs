using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    class nnMath
    {
        // ------------------------------------------------------------ Methoden ------------------------------------------------------------ //

        public nnMath() { }
        public double[] ActivationFunction(double[] inputs, bool activationType) {

            if (activationType == false){
                // ReLu Funktion 
                return inputs.Select(x => Math.Max(0, x)).ToArray();
                
            }
            else{
                // Sigmoid-Funktion
                return inputs.Select(x => 1.0 / (1.0 + Math.Exp(-x))).ToArray();  
            }
            
        }
        public double[] MatrixMult(double[,] matrix, int rows, double[] vector)
        {
            // Dimensionen prüfen
            int cols = matrix.GetLength(1); // Anzahl der Spalten in der Matrix
            if (cols != vector.Length) 
            {
                throw new ArgumentException("Die Dimensionen der Matrix und des Vektors stimmen nicht überein.");
            }

            // Ergebnis-Vektor initialisieren
            double[] result = new double[rows]; // Die Länge des Ergebnisses entspricht der Anzahl der Zeilen (rows) der Matrix.

            // Matrix-Vektor-Multiplikation
            for (int i = 0; i < rows; i++) // Äußere Schleife: Geht durch jede Zeile der Matrix.
            {
                double sum = 0.0;
                for (int j = 0; j < cols; j++) // Innere Schleife: Berechnet das Skalarprodukt der aktuellen Zeile mit dem Vektor.
                {
                    sum += matrix[i, j] * vector[j];
                }
                result[i] = sum;
            }

            return result;
        } 
        public double[] ForwardPass(double[] inputs, double[,] weights, double[] biases, Func<double, double> activationFunction)
        {
            int numNeurons = weights.GetLength(0);
            double[] outputs = new double[numNeurons];

            for (int i = 0; i < numNeurons; i++)
            {
                double sum = biases[i];
                for(int j = 0; j < inputs.Length; j++)
                {
                    sum += weights[i, j] * inputs[j];
                }
                outputs[i] = activationFunction(sum);
            }
            return (outputs);
        }

        public void UpdateWeights(double[] errors, double[] activations, double[,] weights, double learningRate)
        {
            for (int i = 0; i < weights.GetLength(0); i++)
            {
                for (int j = 0; j < weights.GetLength(1); j++)
                {
                    weights[i, j] += learningRate * errors[i] * activations[j];
                }
            }
        }
        public void UpdateBiases(double[] errors, double[] biases, double learningRate)
        {
            for (int i = 0; i < biases.Length; i++)
            {
                biases[i] += learningRate * errors[i];
            }
        }
        public double[] CalculateErrors(double[] targets, double[] outputs)
        {
            if (targets.Length != outputs.Length)
            {
                throw new ArgumentException("Die Dimensionen von Targets und Outputs stimmen nicht überein.");
            }

            return targets.Zip(outputs, (t, o) => t - o).ToArray();
        }
        public double[] BackpropagateErrors(double[] outputErrors, double[,] weightsHO, double[] hiddenOutputs, bool siegmoidReLu)
        {
            int hiddenNodes = weightsHO.GetLength(0);
            double[] hiddenErrors = new double[hiddenNodes];

            for (int i = 0; i < hiddenNodes; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < outputErrors.Length; j++)
                {
                    sum += weightsHO[i, j] * outputErrors[j];
                }
                if (siegmoidReLu)
                { // Aktivierungsfunktion der Sigmoid
                    hiddenErrors[i] = sum * hiddenOutputs[i] * (1 - hiddenOutputs[i]);
                } else {
                    if (hiddenErrors[i] > 0)
                    {
                        hiddenErrors[i] = sum * 1.0; // Ableitung ist 1
                    } else {
                        hiddenErrors[i] = sum * 0.0; // Ableitung ist 0
                    }
                }

            }

            Console.WriteLine($"Hidden Errors: {string.Join(", ", hiddenErrors)}");
            return hiddenErrors;
        }



        // ------------------------------------------------------------ Ableitungen ------------------------------------------------------------ //

        public double SigmoidDerivative(double x)
        {
            double sigmoid = 1.0 / (1.0 + Math.Exp(-x));
            return sigmoid * (1 - sigmoid);
        }

        public double ReLUDerivative(double x)
        {
            return x > 0 ? 1.0 : 0.0;
        }


    }
}
