import ptychoSampling.reconstruction.forwardmodel_t
import ptychoSampling.reconstruction.lossfn_t
import ptychoSampling.reconstruction.optimization_t

OPTIONS = {"forward models":
               {"farfield": ptychoSampling.reconstruction.forwardmodel_t.FarfieldForwardModelT,
                "nearfield": ptychoSampling.reconstruction.forwardmodel_t.NearfieldForwardModelT,
                "bragg": ptychoSampling.reconstruction.forwardmodel_t.BraggPtychoForwardModelT},

           "loss functions":
               {"least_squared": ptychoSampling.reconstruction.lossfn_t.LeastSquaredLossT,
                "gaussian": ptychoSampling.reconstruction.lossfn_t.LeastSquaredLossT,
                "poisson_log_likelihood": ptychoSampling.reconstruction.lossfn_t.PoissonLogLikelihoodLossT,
                "poisson": ptychoSampling.reconstruction.lossfn_t.PoissonLogLikelihoodLossT,
                "poisson_surrogate": ptychoSampling.reconstruction.lossfn_t.PoissonLogLikelihoodSurrogateLossT,
                "intensity_least_squared": ptychoSampling.reconstruction.lossfn_t.IntensityLeastSquaredLossT,
                "counting_model": ptychoSampling.reconstruction.lossfn_t.CountingModelLossT},

           "tf_optimization_methods": {"adam": ptychoSampling.reconstruction.optimization_t.AdamOptimizer,
                                       "gradient": ptychoSampling.reconstruction.optimization_t.GradientDescentOptimizer,
                                       "momentum": ptychoSampling.reconstruction.optimization_t.MomentumOptimizer
                                       }}

           #"optimization_methods": {"adam": ptychoSampling.reconstruction.optimization_t.getAdamOptimizer,
           #                         "custom": ptychoSampling.reconstruction.optimization_t.getOptimizer}}