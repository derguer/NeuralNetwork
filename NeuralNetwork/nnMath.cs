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
        public double[] ActivationFunction(double[] inputs) {

            return inputs.Select(x => 1.0 / (1.0 + Math.Exp(-x))).ToArray();
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


        public double Sigmoid(double x) {

            return 1.0 / (1.0 + Math.Exp(-x));
        }
        
    }
}
