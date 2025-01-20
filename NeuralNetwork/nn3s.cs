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
        

        

        // ------------------------------------------------------------ Methoden ------------------------------------------------------------ //
        public nn3s(int inputNodes, int hiddenNode, int outputNode) {

            InputNodes = inputNodes;
            HiddenNodes = hiddenNode;
            OutputNodes = outputNode;
            CreateWeightMatrizes(); // Initialisiere die Gewichtsmatrizen


        }

        public void QueryNN(double[] inputs)
        {
            nnMath nnMathO = new nnMath();

            if (inputs.Length != _inodes)
            {
                throw new ArgumentException($"Die Eingabedaten müssen {_inodes} Werte enthalten.");
            }

            _hidden_inputs = new double[_hnodes];
            _hidden_inputs = nnMathO.MatrixMult(wih, _hnodes, inputs);
            Console.WriteLine($"_hidden_inputs: {string.Join(", ", _hidden_inputs)}");

            _hidden_outputs = new double[_hnodes];
            _hidden_outputs = nnMathO.ActivationFunction(_hidden_inputs);

            _final_inputs = new double[_onodes];
            _final_inputs = nnMathO.MatrixMult(who, _onodes, _hidden_outputs);

            _final_outputs = new double[_onodes]; // Dimension korrigiert
            _final_outputs = nnMathO.ActivationFunction(_final_inputs);


            Console.WriteLine($"_hidden_inputs: {string.Join(", ", _hidden_inputs)}");
            Console.WriteLine($"_hidden_outputs: {string.Join(", ", _hidden_outputs)}");
            Console.WriteLine($"_final_inputs: {string.Join(", ", _final_inputs)}");
            Console.WriteLine($"_final_outputs: {string.Join(", ", _final_outputs)}");

        }

        private void CreateWeightMatrizes()
        {
            wih = new double[_inodes, _hnodes];
            who = new double[_hnodes, _onodes];

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

        // ------------------------------------------------------------ Properties ------------------------------------------------------------ //

        public int InputNodes { get { return _inodes; } private set{

                if (value > 0)
                {
                    _inodes = value;
                }
                else {
                    throw new Exception("Input Layer muss größer als 0 sein.");
                }; 
            }
        } 
        public int HiddenNodes { get{return _hnodes; } private set {
                if (value > 0)
                {
                    _hnodes = value;
                }
                else {
                    throw new Exception("Hidden Layer muss größer als 0 sein.");
                }; 
            }
        }
        public int OutputNodes { get { return _onodes; } private set{
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

    }
}
