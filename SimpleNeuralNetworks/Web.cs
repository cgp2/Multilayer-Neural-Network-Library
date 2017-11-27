using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleNeuralNetwork
{
    public class SimpleNetwork
    {
        private List<ILayers> layersList = new List<ILayers>();
        public int layersCount = 0;
        double al = 0.01;
        public double eps = 100;

        public SimpleNetwork()
        {
        }


        public void ClearDlt()
        {
            foreach (ILayers lr in layersList)
                foreach (INeurons nr in lr.NeuronsList)
                    foreach (Link lk in nr.IncomingLinksList)
                        lk.dlt = 0;
        }

        public void TeachBackPropagation(List<List<string>> teacher, int batchsize)
        {
            Random rd = new Random();
            int epochCount = 0;

            for (int z = 0; z < 1; z++)
            {
                while (eps > 0.0001)
                {
                    epochCount++;

                    System.Diagnostics.Stopwatch timer = new System.Diagnostics.Stopwatch();
                    timer.Start();
                    for (int v = 0; v < teacher.Count / batchsize; v++)
                    {
                        List<List<string>> rd_t = new List<List<string>>();

                        for (int i = 0; i < batchsize; i++)
                        {
                            int p1 = rd.Next(0, teacher.Count);
                            rd_t.Add(teacher[p1]);
                        }


                        eps = 0;
                        for (int i = 0; i < rd_t.Count; i++)
                        {
                            for (int j = 1; j < rd_t[i].Count; j++)
                            {

                                //Считываем изображение
                                List<double> input = new List<double>();
                                string[] sr = rd_t[i][j].Split(' ');
                                for (int l = 0; l < sr.Length; l++)
                                    input.Add(double.Parse(sr[l]));
                                //Записываем все результаты нейронов
                                List<List<double>> results = new List<List<double>>();
                                results.Add(layersList[0].Result(input));

                                for (int l = 1; l < layersCount; l++)
                                    results.Add(layersList[l].Result(results[l - 1]));

                                //Эталонный результат
                                string[] stndrt = rd_t[i][0].Split(' ');

                                var mistake = CalculateMistakeBackProp(stndrt);

                                //Изменяем веса всех связей
                               RecalculateWeights(mistake);
                            }
                        }
                    }
                    timer.Stop();

                    eps /= teacher.Count;

                    if (double.IsNaN(eps))
                        eps++;
                }

            }

        }

        public void RecalculateWeights(List<List<double>> mistake)
        {
            double treshhold = Math.Pow(10, -8);
            for (int l = 1; l < layersCount; l++)
            {
                for (int m = 1; m < layersList[l].NeuronsCount; m++)
                {
                    //if (layersList[l].NeuronsList[m].IsDropouted == false)
                    //{
                    for (int k = 0; k < layersList[l].NeuronsList[m].IncomingLinksCount; k++)
                    {


                        //if (double.IsNaN(layersList[l].NeuronsList[m].IncomingLinksList[k].pastGradients))
                        //{
                        //    treshhold++;
                        //}
                        layersList[l].NeuronsList[m].IncomingLinksList[k].pastGradients += Math.Pow(layersList[l].NeuronsList[m].IncomingLinksList[k].dlt, 2);
                        if (k == 0)
                        {
                            layersList[l].NeuronsList[m].IncomingLinksList[k].dlt = al /*/ Math.Sqrt(layersList[l].NeuronsList[m].IncomingLinksList[k].pastGradients + treshhold)*/ * mistake[l - 1][m];
                        }
                        else
                        {
                            layersList[l].NeuronsList[m].IncomingLinksList[k].dlt = al / Math.Sqrt(layersList[l].NeuronsList[m].IncomingLinksList[k].pastGradients + treshhold) * mistake[l - 1][m]
                                * layersList[l].NeuronsList[m].IncomingLinksList[k].Out.Result;
                           
                        }

                            layersList[l].NeuronsList[m].IncomingLinksList[k].Weight += layersList[l].NeuronsList[m].IncomingLinksList[k].dlt;
                        }
                    //}
                }
            }
        }

        public void SetDropout(Random rd, double chanse)
        {
            for (int l = 1; l < layersCount - 1; l++)
            {
                for (int nrn = 1; nrn < layersList[l].NeuronsCount; nrn++)
                {
                    double ch = rd.NextDouble();
                    layersList[l].NeuronsList[nrn].IsDropouted = (ch < chanse) ? true : false;
                }
            }
        }

        public List<List<double>> CalculateMistakeBackProp(string[] stndrt)
        {
            List<List<double>> mistake = new List<List<double>>();

            //Ошибка для последнего слоя
            List<double> lastLayerMistake = new List<double>();
            for (int l = 0; l < layersList[layersCount - 1].NeuronsCount; l++)
            {
                double bt = 0;
                switch (layersList[layersCount - 1].NeuronsList[0].Type)
                {
                    case 1://Сигмоид
                        bt = (double.Parse(stndrt[l]) - layersList[layersCount - 1].NeuronsList[l].Result)
                         * layersList[layersCount - 1].NeuronsList[l].Derivate(layersList[layersCount - 1].NeuronsList[l].Result);
                        eps += 0.5 * Math.Pow(double.Parse(stndrt[l]) - layersList[layersCount - 1].NeuronsList[l].Result, 2);
                        break;
                    case 3://Softmax
                        //bt += -1.0 * Math.Log(layersList[layersCount - 1].NeuronsList[l].Result) * double.Parse(stndrt[l]);
                        bt = -1.0 * (layersList[layersCount - 1].NeuronsList[l].Result - double.Parse(stndrt[l]));
                        eps += -1.0 * Math.Log(layersList[layersCount - 1].NeuronsList[l].Result) * double.Parse(stndrt[l]);
                        break;
                    case 4: //TH
                        bt = Math.Abs(double.Parse(stndrt[l]) - layersList[layersCount - 1].NeuronsList[l].Result)
                         * layersList[layersCount - 1].NeuronsList[l].Derivate(layersList[layersCount - 1].NeuronsList[l].Result);
                        eps += 0.5 * Math.Pow(double.Parse(stndrt[l]) - layersList[layersCount - 1].NeuronsList[l].Result, 2);
                        break;
                }
                lastLayerMistake.Add(bt);
            }
            mistake.Add(lastLayerMistake);

            //Ошибка для остальных слоев, начиная от предпоследнего
            for (int l = layersCount - 2; l > -1; l--)
            {
                List<double> db = new List<double>();
                // Cчитаем сумму входяших ошибок в слой l
                List<double> reverssum = this.ReversSum(l, mistake[mistake.Count - 1]);
                if (l != 0)
                {
                    for (int m = 0; m < layersList[l].NeuronsCount; m++)
                    {
                        if (layersList[l].NeuronsList[m].IsDropouted == false)
                        {
                            //double o = results[l][m] * (1 - results[l][m]) * reverssum[m];
                            double o = reverssum[m] * layersList[l].NeuronsList[m].Derivate(layersList[l].NeuronsList[m].Result);
                            db.Add(o);
                        }
                        else
                            db.Add(0);
                    }
                    mistake.Add(db);
                }
            }
            //переворачиваем массив ошибок
            mistake.Reverse();

            return mistake;
        }

        private List<List<double>> CalculateMistakeBFGS(string[] stndrt)
        {
            List<List<double>> hessian = new List<List<double>>();
            List<List<double>> mistake = new List<List<double>>();

            //Ошибка для последнего слоя
            List<double> lastLayerMistake = new List<double>();
            for (int l = 0; l < layersList[layersCount - 1].NeuronsCount; l++)
            {
                double bt = 0;
                switch (layersList[layersCount - 1].NeuronsList[0].Type)
                {
                    case 1://Сигмоид
                        bt = (double.Parse(stndrt[l]) - layersList[layersCount - 1].NeuronsList[l].Result)
                         * layersList[layersCount - 1].NeuronsList[l].Derivate(layersList[layersCount - 1].NeuronsList[l].Result);
                        eps += 0.5 * Math.Pow(double.Parse(stndrt[l]) - layersList[layersCount - 1].NeuronsList[l].Result, 2);
                        break;
                    case 3://Softmax
                        //bt += -1.0 * Math.Log(layersList[layersCount - 1].NeuronsList[l].Result) * double.Parse(stndrt[l]);
                        bt = -1.0 * (layersList[layersCount - 1].NeuronsList[l].Result - double.Parse(stndrt[l]));
                        eps += -1.0 * Math.Log(layersList[layersCount - 1].NeuronsList[l].Result) * double.Parse(stndrt[l]);
                        break;
                    case 4: //TH
                        bt = Math.Abs(double.Parse(stndrt[l]) - layersList[layersCount - 1].NeuronsList[l].Result)
                         * layersList[layersCount - 1].NeuronsList[l].Derivate(layersList[layersCount - 1].NeuronsList[l].Result);
                        eps += 0.5 * Math.Pow(double.Parse(stndrt[l]) - layersList[layersCount - 1].NeuronsList[l].Result, 2);
                        break;
                }
                lastLayerMistake.Add(bt);


            }
            mistake.Add(lastLayerMistake);

            //Ошибка для остальных слоев, начиная от предпоследнего
            for (int l = layersCount - 2; l > -1; l--)
            {
                List<double> db = new List<double>();
                // Cчитаем сумму входяших ошибок в слой l
                List<double> reverssum = this.ReversSum(l, mistake[mistake.Count - 1]);
                if (l != 0)
                {
                    for (int m = 0; m < layersList[l].NeuronsCount; m++)
                    {
                        if (layersList[l].NeuronsList[m].IsDropouted == false)
                        {
                            //double o = results[l][m] * (1 - results[l][m]) * reverssum[m];
                            double o = reverssum[m] * layersList[l].NeuronsList[m].Derivate(layersList[l].NeuronsList[m].Result);
                            db.Add(o);
                        }
                        else
                            db.Add(0);
                    }
                    mistake.Add(db);
                }
            }
            //переворачиваем массив ошибок
            mistake.Reverse();

            return mistake;
        }

        private List<double> ReversSum(int n, List<double> input)
        {
            List<double> resullt = new List<double>();

            //Проходим по всем нейронам слоя n
            for (int i = 0; i < layersList[n].NeuronsCount; i++)
            {
                if (layersList[n].NeuronsList[i].IsDropouted == false)
                {
                    double sum = 0;
                    //Проходим по всем нейронам слоя n+1
                    if (n == layersCount - 2)
                    {
                        for (int j = 0; j < layersList[n + 1].NeuronsCount; j++)
                        {
                            if (layersList[n + 1].NeuronsList[j].IsDropouted == false)
                            {
                                //у каждого нейрона слоя n+1 берем i-ую связь и считаем их взвешенную сумму
                                sum += input[j] * layersList[n + 1].NeuronsList[j].IncomingLinksList[i].Weight;
                            }
                        }
                    }
                    else
                    {
                        for (int j = 1; j < layersList[n + 1].NeuronsCount; j++)
                        {
                            if (layersList[n + 1].NeuronsList[j].IsDropouted == false)
                            {
                                //у каждого нейрона слоя n+1 берем i-ую связь и считаем их взвешенную сумму
                                sum += input[j] * layersList[n + 1].NeuronsList[j].IncomingLinksList[i].Weight;
                            }
                        }
                    }
                    resullt.Add(sum);
                }
                else
                    resullt.Add(0);
            }

            return resullt;
        }

        public List<double> Result(List<double> input)
        {
            List<double> res = input;

            foreach (ILayers lr in layersList)
                res = lr.Result(res);

            return res;
        }

        public void SetAllLinks()
        {
            for (int i = 1; i < layersCount; i++)
            {
                if (layersList[i].NeuronsList[1].IncomingLinksCount == 0)
                    layersList[i].AddLinks(layersList[i - 1].NeuronsCount);

                if (i == layersCount - 1)
                {
                    for (int j = 0; j < layersList[i].NeuronsCount; j++)
                    {
                        for (int k = 0; k < layersList[i].NeuronsList[j].IncomingLinksCount; k++)
                        {
                            layersList[i].NeuronsList[j].IncomingLinksList[k].Out = layersList[i - 1].NeuronsList[k];
                        }
                    }
                }
                else
                {
                    for (int j = 1; j < layersList[i].NeuronsCount; j++)
                    {
                        for (int k = 0; k < layersList[i].NeuronsList[j].IncomingLinksCount; k++)
                        {
                            layersList[i].NeuronsList[j].IncomingLinksList[k].Out = layersList[i - 1].NeuronsList[k];
                        }
                    }
                }
            }
        }

        public void RandomizeWeights()
        {
            Random rd = new Random();

            for (int i = 1; i < layersCount; i++)
            {
                List<List<double>> t = new List<List<double>>();

                foreach (INeurons nr in layersList[i].NeuronsList)
                {
                    List<double> t1 = new List<double>();

                    foreach (Link lk in nr.IncomingLinksList)
                        t1.Add(RandomDouble(rd, -0.5, 0.5));

                    t.Add(t1);
                }

                layersList[i].SetWeights(t);
            }
        }

        private double RandomDouble(Random rd, double min, double max)
        {
            return rd.NextDouble() * (max - min) + min;
        }

        public void AddInputLayer(int n)
        {
            Layer_Input lr_i = new Layer_Input(n);
            layersList.Add(lr_i);
            layersCount++;
        }

        public void AddSigmoidLayer(int n, int t)
        {
            Layer_Sigmoid lr_s = new Layer_Sigmoid(n, t);
            layersList.Add(lr_s);
            layersCount++;
            SetAllLinks();
        }

        public void AddReluLayer(int n)
        {
            Layer_Relu lr_r = new Layer_Relu(n);
            layersList.Add(lr_r);
            layersCount++;
            SetAllLinks();
        }

        public void AddSoftmaxLayer(int n)
        {
            Layer_Softmax lr_s = new Layer_Softmax(n);
            layersList.Add(lr_s);
            layersCount++;
            SetAllLinks();
        }

        public void AddTHLayer(int n, int t)
        {
            Layer_TH lr_TH = new Layer_TH(n, t);
            layersList.Add(lr_TH);
            layersCount++;
            SetAllLinks();
        }

        public void ReadWeightsFromFile(string path)
        {
            layersList.Clear();
            layersCount = 0;
            string[] input = System.IO.File.ReadAllLines(@path);

            int na = input.Count(i => i == ">") + 1;
            List<string> s = input.ToList();
            int p = 1;

            int m = int.Parse(input[0]);
            this.AddInputLayer(m);

            for (int i = 1; i < na; i++)
            {
                int n = s.IndexOf(">", p + 1);
                int t = int.Parse(s[p + 1]);
                var s1 = s.GetRange(p + 2, n - p - 2);

                switch (t)
                {
                    case 1:
                        if (i == na - 1)
                            this.AddSigmoidLayer(n - p - 2, 2);
                        else
                            this.AddSigmoidLayer(n - p - 2, 1);
                        break;
                    case 2:
                        this.AddReluLayer(n - p - 2);
                        break;
                    case 3:
                        this.AddSoftmaxLayer(n - p - 2);
                        break;
                }

                layersList[i].SetWeights(s1.ToArray());

                p = n + 1;
            }

            SetAllLinks();
        }
        public void SaveToFile(string path)
        {
            System.IO.FileStream fs = new System.IO.FileStream(@path, System.IO.FileMode.Create, System.IO.FileAccess.Write);
            System.IO.StreamWriter sw = new System.IO.StreamWriter(fs);

            sw.WriteLine(Convert.ToString(layersList[0].NeuronsList.Count - 1));

            foreach (ILayers lr in layersList)
            {
                if (lr.Type != 0)
                {
                    sw.WriteLine("<");
                    sw.WriteLine(lr.NeuronsList[0].Type);
                    foreach (INeurons nrn in lr.NeuronsList)
                    {
                        if (nrn.IncomingLinksCount != 0)
                        {
                            for (int i = 0; i < nrn.IncomingLinksCount; i++)
                            {
                                if (i != nrn.IncomingLinksCount - 1)
                                    sw.Write(nrn.IncomingLinksList[i].Weight + " ");
                                else
                                    sw.Write(nrn.IncomingLinksList[i].Weight);
                            }
                            sw.WriteLine();
                        }
                    }
                    sw.WriteLine(">");
                }

            }


            sw.Close();
        }

        public List<ILayers> LayersList
        {
            get
            {
                return layersList;
            }
            set
            {
                layersList = value;
            }
        }
    }
}
