using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleNeuralNetwork
{
    public interface INeurons
    {
        int IncomingLinksCount { get; set; }
        int Type { get; set; } // 0 - входной нейрон, 1 - сигмоид, 2 - Relu, 3 - Softmax, 4 - TH
        double Result { get; set; }
        double Sum { get; }
        int Id { get; }
        List<Link> IncomingLinksList { get; set; }
        bool IsDropouted { get; set; }

        void SetLinks(List<double> w);
        void AddIncomingLinks(int n);
        double Res(List<double> input);
        double Derivate(double x);
    }
}
