﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleNeuralNetwork
{ 
    public class Link
    {
        public double Weight, dlt = 0, pastGradients  = 0.000000001;
        public INeurons In, Out;
        
        public Link(double w, INeurons n)
        {
            Weight = w;
            In = n;
        }
    }
}
