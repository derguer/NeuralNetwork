using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    class nn3s
    {

        double[,] wih, who; // weights input-hidden, weights hidden-output
        /// <summary>
        /// Beispiel: wih[i, j] ist das Gewicht der Verbindung zwischen dem i-ten Eingabeknoten und dem j-ten versteckten Knoten.
        /// Beispiel: who[i, j] ist das Gewicht der Verbindung zwischen dem i-ten versteckten Knoten und dem j-ten Ausgabeknoten.
        /// </summary>

        private double[] _hidden_inputs; // summierten Eingabewerte
        private double[] _hidden_outputs; // aktivierten Ausgabewerte der versteckten Schicht
        private double[] _final_inputs; // summierten Eingabewerte
        private double[] _final_outputs; // aktivierten Ausgabewerte der Ausgabeschicht
        private double[] _hidden_biases;
        private double[] _output_biases;
        private double[] _output_errors; // Fehler im Output-Layer
        private double[] _hidden_errors;



        private double _lernrate;

        private int _inodes; // (Input-Nodes)
        /// <summary>
        /// Diese repräsentieren die Eingangsdaten, die in das Netz eingespeist werden. 
        /// Beispiel: Wenn du ein Bild mit 28x28 Pixeln analysieren möchtest, wären inodes = 784, da jedes Pixel ein Eingabewert ist.
        /// </summary>

        private int _hnodes; // (Hidden-Nodes)
        /// <summary>
        /// Diese Knoten befinden sich in den Zwischenschichten zwischen der Eingabe- und der Ausgabeschicht.
        /// 
        /// </summary>

        private int _onodes; // (Output-Nodes)
        /// <summary>
        /// Diese repräsentieren die Ergebnisse des Netzes.
        /// Beispiel: Bei einer Klassifikationsaufgabe mit drei Klassen wären onodes = 3, da jede Klasse durch einen Knoten dargestellt wird.
        /// </summary>
        /// 
        private bool _siegmoidReLu; // Variable für Sigmoid oder ReLu Funktion 




        // ------------------------------------------------------------ Methoden ------------------------------------------------------------ //
        public nn3s(int inputNodes, int hiddenNode, int outputNode, double lernrate, bool funktion) {

            InputNodes = inputNodes;
            HiddenNodes = hiddenNode;
            OutputNodes = outputNode;
            Lernrate = lernrate;
            Sigmoid_ReLu = funktion;
            CreateWeightMatrizes(); // Initialisiere die Gewichtsmatrizen
        }

        public void QueryNN(double[] inputs, bool sigmoid_ReLu)
        {
            nnMath nnMathO = new nnMath();

            if (inputs.Length != _inodes)
            {
                throw new ArgumentException($"Die Eingabedaten müssen {_inodes} Werte enthalten.");
            }

            _hidden_inputs = nnMathO.MatrixMult(wih, _hnodes, inputs);
            Console.WriteLine($"_hidden_inputs: {string.Join(", ", _hidden_inputs)}");

            _hidden_outputs = nnMathO.ActivationFunction(_hidden_inputs, sigmoid_ReLu);

            _final_inputs = nnMathO.MatrixMult(who, _onodes, _hidden_outputs);
            _final_outputs = nnMathO.ActivationFunction(_final_inputs, sigmoid_ReLu);

            // Dummy-Initialisierung für Output_errors
            if (_output_errors == null)
            {
                _output_errors = new double[_onodes];
                for (int i = 0; i < _output_errors.Length; i++)
                {
                    _output_errors[i] = 0; // Dummy-Wert
                }
            }

        }


        private void CreateWeightMatrizes()
        {
            wih = new double[_inodes, _hnodes];
            who = new double[_hnodes, _onodes];
            _hidden_biases = new double[_hnodes];
            _output_biases = new double[_onodes];

            wih[0, 0] = 0.9;
            wih[1, 0] = 0.3;
            wih[2, 0] = 0.4;
            wih[0, 1] = 0.2;
            wih[1, 1] = 0.8;
            wih[2, 1] = 0.2;
            wih[0, 2] = 0.1;
            wih[1, 2] = 0.5;
            wih[2, 2] = 0.6;

            who[0, 0] = 0.3;
            who[1, 0] = 0.7;
            who[2, 0] = 0.5;
            who[0, 1] = 0.6;
            who[1, 1] = 0.5;
            who[2, 1] = 0.2;
            who[0, 2] = 0.8;
            who[1, 2] = 0.1;
            who[2, 2] = 0.9;

            /*for (int j = 0; j < _hnodes; j++)
                for (int i = 0; i < _inodes; i++)
                {
                    System.Random weight_ih = new System.Random();
                    wih[i, j] = weight_ih.NextDouble() - 0.5;
                    //Console.WriteLine("i: " + i + ", j: " + j + ", w: " + wih[i, j].ToString());
                }
            for (int j = 0; j < _onodes; j++)
                for (int i = 0; i < _hnodes; i++)
                {
                    System.Random weight_ho = new System.Random();
                    who[i, j] = weight_ho.NextDouble() - 0.5;
                    //Console.WriteLine("i: " + i + ", j: " + j + ", w: " + who[i, j].ToString());
                }*/
        }

        public void Train(double[] inputs, double[] targets)
        {
            if (inputs.Length != _inodes || targets.Length != _onodes)
            {
                throw new ArgumentException("Die Dimensionen von Inputs oder Targets stimmen nicht mit den Knoten überein.");
            }

            if (wih == null || who == null || _hidden_biases == null || _output_biases == null)
            {
                throw new InvalidOperationException("Gewichtsmatrizen oder Biases wurden nicht korrekt initialisiert.");
            }

            nnMath math = new nnMath();

            // 1. Vorwärtsdurchlauf: Berechnung der Aktivierungen
            _hidden_inputs = math.MatrixMult(wih, _hnodes, inputs);
            _hidden_outputs = math.ActivationFunction(_hidden_inputs, false); // Sigmoid

            _final_inputs = math.MatrixMult(who, _onodes, _hidden_outputs);
            _final_outputs = math.ActivationFunction(_final_inputs, false); // Sigmoid

            // 2. Fehlerberechnung
            _output_errors = math.CalculateErrors(targets, _final_outputs);
            _hidden_errors = math.BackpropagateErrors(_output_errors, who, _hidden_inputs, _siegmoidReLu);

            // 3. Aktualisierung der Gewichte und Biases
            math.UpdateWeights(_output_errors, _hidden_outputs, who, _lernrate); // Hidden → Output
            math.UpdateBiases(_output_errors, _output_biases, _lernrate); // Biases für Output-Layer

            math.UpdateWeights(_hidden_errors, inputs, wih, _lernrate); // Input → Hidden
            math.UpdateBiases(_hidden_errors, _hidden_biases, _lernrate); // Biases für Hidden-Layer
        }
        public double[] GetRowFromWHO(int rowIndex)
        {
            if (rowIndex < 0 || rowIndex >= who.GetLength(0))
            {
                throw new ArgumentOutOfRangeException(nameof(rowIndex), "Index liegt außerhalb der gültigen Bereichs.");
            }

            double[] row = new double[who.GetLength(1)];
            for (int j = 0; j < who.GetLength(1); j++)
            {
                row[j] = who[rowIndex, j];
            }
            return row;
        }
        private double[] GetRowFromWHI(double[,] matrix, int rowIndex)
        {
            if (rowIndex < 0 || rowIndex >= matrix.GetLength(0))
            {
                throw new ArgumentOutOfRangeException(nameof(rowIndex), "Index liegt außerhalb des gültigen Bereichs.");
            }

            double[] row = new double[matrix.GetLength(1)];
            for (int j = 0; j < matrix.GetLength(1); j++)
            {
                row[j] = matrix[rowIndex, j];
            }
            return row;
        }
        public string GetFormattedOutputErrors()
        {
            if (Output_errors == null)
            {
                throw new InvalidOperationException("Output_errors ist nicht initialisiert. Rufen Sie zuerst die Train-Methode auf.");
            }

            // Formatieren als Zeilen
            return string.Join("\n", _output_errors.Select(e => $"{e:0.##}"));

        }


        // ------------------------------------------------------------ Properties ------------------------------------------------------------ //

        public int InputNodes { get { return _inodes; } private set {

                if (value > 0)
                {
                    _inodes = value;
                }
                else {
                    throw new Exception("Input Layer muss größer als 0 sein.");
                };
            }
        }
        public int HiddenNodes { get { return _hnodes; } private set {
                if (value > 0)
                {
                    _hnodes = value;
                }
                else {
                    throw new Exception("Hidden Layer muss größer als 0 sein.");
                };
            }
        }
        public int OutputNodes { get { return _onodes; } private set {
                if (value > 0)
                {
                    _onodes = value;
                }
                else {
                    throw new Exception("Output Layer muss größer als 0 sein.");
                };
            }
        }
        public double[] Hidden_inputs { get { return _hidden_inputs; } }
        public double[] Hidden_outputs { get { return _hidden_outputs; } }
        public double[] Final_inputs { get { return _final_inputs; } }
        public double[] Final_outputs { get { return _final_outputs; } }

        public double Lernrate
        {
            get { return _lernrate; }
            private set
            {
                if (value > 0)
                {
                    _lernrate = value;
                };
            }
        }
        
        public string GetFormattedRowFromWHO(int rowIndex)
        {
            return string.Join(", ", GetRowFromWHO(rowIndex).Select(w => $"{w:0.##}"));
        }
        public string GetFormattedRowFromWIH(int rowIndex)
        {
            return string.Join(", ", GetRowFromWHI(wih, rowIndex).Select(w => $"{w:0.##}"));
        }
        public double[] GetOutputErrors()
        {
            return Output_errors; // Gibt den Fehlervektor zurück
        }

        public double[] Output_errors => _output_errors; // Getter für ErrorOutput 
        public double[] Hidden_errors => _hidden_errors; // Getter für ErrorHidden
        public bool Sigmoid_ReLu { get{ return _siegmoidReLu; }private set {_siegmoidReLu=value; } }



    }

}

