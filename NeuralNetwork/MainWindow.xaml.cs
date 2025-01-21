using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace NeuralNetwork
{
    /// <summary>
    /// Interaktionslogik für MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        
        private int inodes = 3;
        private int hnodes = 3;
        private int onodes = 3;
        private double[] inputs = Array.Empty<double>(); // Dekleration Input-Vektor
        private double[] targets = Array.Empty<double>(); // Dekleration Target-Vektor
        private nn3s nn3SO; // Dekleration der Variable nn3SO ein Objekt der Klasse nn3S
        private double learningRate = 0.1; // Instanzierung der Lernrate of the NN
        private bool siegmoid_ReLU = true; // Bool für andere Funktion statt Sigmoid, für später. 




        public MainWindow()
        {
            InitializeComponent();
        }

        private void inputTextBox_PreviewTextInput(object sender, TextCompositionEventArgs e)
        {
            e.Handled = !int.TryParse(e.Text, out int inodes);
            Console.WriteLine("input : " + inodes);
        }

        private void hiddenTextBox_PreviewTextInput(object sender, TextCompositionEventArgs e)
        {
            e.Handled = !int.TryParse(e.Text, out int hnodes);
        }

        private void outputTextBox_PreviewTextInput(object sender, TextCompositionEventArgs e)
        {
            e.Handled = !int.TryParse(e.Text, out int onodes);
        }
        private void learningRateTextBox_PreviewTextInput(object sender, TextCompositionEventArgs e)
        {
            e.Handled = !int.TryParse(e.Text, out int onodes);
            Console.WriteLine("input : " + learningRate);
        }
        private void sigmoidReLu_Checked(object sender, RoutedEventArgs e)
        {
            if (sigmoidReLu.IsChecked == true)
            {
                siegmoid_ReLU = true;
                Console.WriteLine(" Funktion auf Sigmoid gestellt :D");
            }
            else
            {
                siegmoid_ReLU = false;
                Console.WriteLine(" Funktion auf ReLu gestellt :)");
            }
        }

        private void createButton_Click(object sender, RoutedEventArgs e) => InitializeNetwork();
        private void trainButton_Click(object sender, RoutedEventArgs e) => TrainNetwork();
        private void queryButton_Click(object sender, RoutedEventArgs e) => QueryNetwork();

        private void SetTestValues()
        {
            inputs = new double[inodes];
            targets = new double[onodes];

            inputs[0] = 0.9;
            inputs[1] = 0.1;
            inputs[2] = 0.8;

            targets[0] = 0.9;
            targets[1] = 0.9;
            targets[2] = 0.9;
        }

        private void InitializeNetwork()
        {
            nn3SO = new nn3s(inodes, hnodes, onodes, learningRate, siegmoid_ReLU);
            inputs = new double[inodes];
            targets = new double[onodes];

            trainButton.IsEnabled = true; // Aktiviert den Train-Button
            queryButton.IsEnabled = true; // Aktiviert den Query-Button
        }

        private void TrainNetwork()
        {
            if (nn3SO == null)
            {
                MessageBox.Show("Das neuronale Netzwerk wurde nicht initialisiert.");
                return;
            }

            SetTestValues(); // Testwerte setzen

            if (!ValidateInputs()) return;
            nn3SO.Train(inputs, targets);
            DisplayResults();
        }

        private void QueryNetwork()
        {
            if (nn3SO == null)
            {
                MessageBox.Show("Das neuronale Netzwerk wurde nicht initialisiert.");
                return;
            }

            SetTestValues(); // Testwerte setzen
            if (!ValidateInputs()) return;
            nn3SO.QueryNN(inputs, siegmoid_ReLU);
            DisplayResults();
        }


        public void DisplayResults()
        {
            //networkDataGrid.Items.Clear(); // Alte Einträge löschen
            for (int i = 0; i < inodes; i++)
            {

                string weightIHColumn = nn3SO.GetFormattedRowFromWIH(i);
                string weightHOColumn = nn3SO.GetFormattedRowFromWHO(i);

                nodeRow data = new nodeRow
                {
                    inputValue = inputs[i].ToString(),
                    weightsIH = weightIHColumn,
                    inputHidden = $"{nn3SO.Hidden_inputs[i]:0.##}",
                    outputHidden = $"{nn3SO.Hidden_outputs[i]:0.##}",
                    weightsHO = weightHOColumn,
                    //errorHidden = $"{nn3SO.Hidden_errors[i]:0.##}",
                    errorHidden = nn3SO.Hidden_errors == null? "Nicht verfügbar": $"{nn3SO.Hidden_errors[i]:0.##}", // Wenn Error-Output Null ist wird der Wert 0 gesetzt, weil beim Forward prop, keine Errors berechnet werden. 
                    inputOutput = $"{nn3SO.Final_inputs[i]:0.##}",
                    outputLayer = $"{nn3SO.Final_outputs[i]:0.##}",
                    target = targets[i].ToString(),                   
                    errorOutput = $"{nn3SO.Output_errors[i]:0.##}", // Fehler pro Zeile 
                    IsCurrent = (i == inodes - 1) // Markiere die letzte Zeile als aktuell
                };
                networkDataGrid.Items.Add(data);
            }
        }

        private bool ValidateInputs()
        {
            if (inodes <= 0 || hnodes <= 0 || onodes <= 0 || learningRate <= 0)
            {
                MessageBox.Show("Alle Werte müssen positiv und größer als 0 sein.");
                return false;
            }
            if (inputs.Length != inodes || targets.Length != onodes)
            {
                MessageBox.Show("Die Eingaben und Zielwerte müssen die richtige Dimension haben.");
                return false;
            }
            return true;
        }    
    }

    public class nodeRow
    {
        public string inputValue { get; set; }
        public string weightsIH { get; set; }
        public string inputHidden { get; set; }
        public string outputHidden { get; set; }
        public string weightsHO { get; set; }
        public string errorHidden { get; set; }
        public string inputOutput { get; set; }
        public string outputLayer { get; set; }
        public string target { get; set; }
        public string errorOutput { get; set; }
        // Neue Eigenschaft zur Markierung
        public bool IsCurrent { get; set; } // True, wenn diese Zeile hervorgehoben werden soll

    }
}

