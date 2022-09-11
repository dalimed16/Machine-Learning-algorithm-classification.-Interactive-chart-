# Get this figure: fig = py.get_figure("https://plotly.com/~dalimed16/3/")
# Get this figure's data: data = py.get_figure("https://plotly.com/~dalimed16/3/").get_data()
# Add data to this figure: py.plot(Data([Scatter(x=[1, 2], y=[2, 3])]), filename ="Plot 1 copy", fileopt="extend")

# Get figure documentation: https://plotly.com/python/get-requests/
# Add data documentation: https://plotly.com/python/file-options/

# If you're using unicode in your file, you may need to specify the encoding.
# You can reproduce this figure in Python with the following code!

# Learn about API authentication here: https://plotly.com/python/getting-started
# Find your api_key here: https://plotly.com/settings/api

import plotly.plotly as py
from plotly.graph_objs import *
py.sign_in('username', 'api_key')
trace1 = {
  "meta": {"columnNames": {
      "ids": "data.0.ids", 
      "labels": "data.0.labels", 
      "parents": "data.0.parents", 
      "customdata": "data.0.customdata"
    }}, 
  "sort": True, 
  "type": "sunburst", 
  "level": "Machine Learning", 
  "idssrc": "dalimed16:2:5bc57a", 
  "ids": ["Machine Learning", "Supervised", "Regression", "Linear Regression", "Multivariate Adaptive Regression Splines (MARS)", "Locally Weighted Scatterplot Smoothing (LOWESS)", "Support Vector Regression (SVR)", "Decision Tree Regression", "Random Forest Regression", "K-Nearest Neighbors Regression (KNN)", "Classification", "Logistic Regression", "Naive Bayes", "Support Vector Machines (SVM)", "Decision Tree Classification", "Random Forest Classification", "Adaptive Boosting (AdaBoost)", "Gradient Boosted Trees", "Extreme Gradient Boosting (XGBoost)", "K-Nearest Neighbors Classification (KNN)", "Dimensionality Reduction ", "Linear Discriminant Analysis (LDA)", "Unsupervised", "Clustering", "K-Means", "Gaussian Mixture Models (GMM)", "Hierarchical Agglomerative Clustering (HAC)", "Density-Based Spatial Clustering of Applications with Noise (DBSCAN)", "Association", "Apriori", "Dimensionality Reduction", "Uniform Manifold Approximation and Projection (UMAP)", "Principal Component Analysis (PCA)", "Multidimensional Scaling (MDS)", "Isomap Embedding", "t-Distributed Stochastic Neighbor Embedding (t-SNE)", "Locally Linear Embedding (LLE)", "Neural Networks", "Feed Forward Neural Networks", "Feed Forward (FF)", "Deep Feed Forward (DFF)", "Recurrent Neural Networks", "Recurrent Neural Network (RNN)", "Long Short Term Memory (LSTM)", "Gated Reccurent Unit (GRU)", "Convolutional Neural Networks", "Deep Convolutional Network (DCN)", "Deconvolutional Network (DN)", "Auto Encoders", "Auto Encoder (AE)", "Variational Auto Encoder (VAE)", "Denoising Auto Encoder (DAE)", "Sparse Auto Encoder (SAE)", "Generative Adversarial Networks", "Generative Adversarial Network (GAN)", "Conditional GAN (cGAN)", "Deep Convolutional GAN (DCGAN)", "Cycle GAN", "Wasserstein GAN (WGAN)", "Semi-Supervised", "Self Training Classifier", "Label Spreading", "Label Propagation", "Reinforcement", "Monte Carlo Methods", "Temporal-Difference (TD)", "Policy Gradient", "Proximal Policy Optimization (PPO)", "SARSA (State-Action-Reward-State-Action)", "Q-Learning", "Deep Q Neural Network (DQN)", "Others", "Probabilistic Graphical Models", "Bayesian Belief Networks"], 
  "marker": {
    "meta": {"columnNames": {"colors": "data.0.marker.colors"}}, 
    "colorssrc": "dalimed16:2:226e29", 
    "colors": ["white", "#EF553B", "rgb(251,128,114)", "rgb(251,128,114)", "rgb(251,128,114)", "rgb(251,128,114)", "rgb(251,128,114)", "rgb(251,128,114)", "rgb(251,128,114)", "rgb(251,128,114)", "#ba2020", "#ba2020", "#ba2020", "#ba2020", "#ba2020", "#ba2020", "#ba2020", "#ba2020", "#ba2020", "#ba2020", "rgb(252,195,195)", "rgb(252,195,195)", "#00CC96", "rgb(204,235,197)", "rgb(204,235,197)", "rgb(204,235,197)", "rgb(204,235,197)", "rgb(204,235,197)", "rgb(141,211,199)", "rgb(141,211,199)", "#a3e897", "#a3e897", "#a3e897", "#a3e897", "#a3e897", "#a3e897", "#a3e897", "#ffe600", "#faf693", "#faf693", "#faf693", "#fff16b", "#fff16b", "#fff16b", "#fff16b", "#fcffc2", "#fcffc2", "#fcffc2", "#ffd857", "#ffd857", "#ffd857", "#ffd857", "#ffd857", "#ffb300", "#ffb300", "#ffb300", "#ffb300", "#ffb300", "#ffb300", "#ff5ac3", "#ff5ac3", "#ff5ac3", "#ff5ac3", "#45abff", "#45abff", "#45abff", "#45abff", "#45abff", "#45abff", "#45abff", "#45abff", "#AB63FA", "#AB63FA", "#AB63FA"]
  }, 
  "maxdepth": 3, 
  "hoverinfo": "label+text", 
  "labelssrc": "dalimed16:2:695b0b", 
  "labels": ["Machine<br>Learning", "Supervised", "Regression", "Linear Regression", "Multivariate<br>Adaptive Regression<br>Splines (MARS)", "Locally Weighted<br>Scatterplot<br>Smoothing (LOWESS)", "Support Vector<br>Regression (SVR)", "Decision Tree<br>Regression (CART)", "Random Forest<br>Regression", "K-Nearest Neighbors<br>Regression (KNN)", "Classification", "Logistic Regression", "Naive Bayes", "Support Vector<br>Machines (SVM)", "Decision Tree<br>Classification<br>(CART)", "Random Forest<br>Classification", "Adaptive Boosting<br>(AdaBoost)", "Gradient Boosted<br>Trees", "Extreme Gradient<br>Boosting (XGBoost)", "K-Nearest Neighbors<br>Classification (KNN)", "Dimensionality<br>Reduction", "Linear Discriminant<br>Analysis (LDA)", "Unsupervised", "Clustering", "K-Means", "Gaussian Mixture<br>Models (GMM)", "Hierarchical<br>Agglomerative<br>Clustering (HAC)", "Density-Based<br>Spatial Clustering<br>of Applications with<br>Noise (DBSCAN)", "Association", "Apriori", "Dimensionality<br>Reduction", "Uniform Manifold<br>Approximation and<br>Projection (UMAP)", "Principal Component<br>Analysis (PCA)", "Multidimensional<br>Scaling (MDS)", "Isomap Embedding", "t-Distributed<br>Stochastic Neighbor<br>Embedding (t-SNE)", "Locally Linear<br>Embedding (LLE)", "Neural Networks", "Feed Forward Neural<br>Networks", "Feed Forward (FF)", "Deep Feed Forward<br>(DFF)", "Recurrent Neural<br>Networks", "Recurrent Neural<br>Network (RNN)", "Long Short Term<br>Memory (LSTM)", "Gated Reccurent Unit<br>(GRU)", "Convolutional Neural<br>Networks", "Deep Convolutional<br>Network (DCN)", "Transposed<br>Convolutional<br>Network", "Auto Encoders", "Undercomplete Auto<br>Encoder (AE)", "Variational Auto<br>Encoder (VAE)", "Denoising Auto<br>Encoder (DAE)", "Sparse Auto Encoder<br>(SAE)", "Generative<br>Adversarial Networks", "Generative<br>Adversarial Network<br>(GAN)", "Conditional GAN<br>(cGAN)", "Deep Conolutional<br>GAN (DCGAN)", "Cycle GAN", "Wasserstein GAN<br>(WGAN)", "Semi-Supervised", "Self Training<br>Classifier", "Label Spreading", "Label Propagation", "Reinforcement", "Monte Carlo Methods", "Temporal-Difference<br>(TD)", "Policy Gradient", "Proximal Policy<br>Optimization (PPO)", "SARSA (State-Action-<br>Reward-State-Action)", "Q-Learning", "Deep Q Neural<br>Network (DQN)", "Others", "Probabilistic<br>Graphical Models", "Bayesian Belief<br>Networks (BBN)"], 
  "parentssrc": "dalimed16:2:3c607c", 
  "parents": [None, "Machine Learning", "Supervised", "Regression", "Regression", "Regression", "Regression", "Regression", "Regression", "Regression", "Supervised", "Classification", "Classification", "Classification", "Classification", "Classification", "Classification", "Classification", "Classification", "Classification", "Supervised", "Dimensionality Reduction ", "Machine Learning", "Unsupervised", "Clustering", "Clustering", "Clustering", "Clustering", "Unsupervised", "Association", "Unsupervised", "Dimensionality Reduction", "Dimensionality Reduction", "Dimensionality Reduction", "Dimensionality Reduction", "Dimensionality Reduction", "Dimensionality Reduction", "Machine Learning", "Neural Networks", "Feed Forward Neural Networks", "Feed Forward Neural Networks", "Neural Networks", "Recurrent Neural Networks", "Recurrent Neural Networks", "Recurrent Neural Networks", "Neural Networks", "Convolutional Neural Networks", "Convolutional Neural Networks", "Neural Networks", "Auto Encoders", "Auto Encoders", "Auto Encoders", "Auto Encoders", "Neural Networks", "Generative Adversarial Networks", "Generative Adversarial Networks", "Generative Adversarial Networks", "Generative Adversarial Networks", "Generative Adversarial Networks", "Machine Learning", "Semi-Supervised", "Semi-Supervised", "Semi-Supervised", "Machine Learning", "Reinforcement", "Reinforcement", "Reinforcement", "Reinforcement", "Reinforcement", "Reinforcement", "Reinforcement", "Machine Learning", "Others", "Probabilistic Graphical Models"], 
  "customdatasrc": "dalimed16:2:5e8303", 
  "customdata": [" ", " ", " ", "Article on: https://solclover.com", "Article on: https://solclover.com", "Article on: https://solclover.com", "Article on: https://solclover.com", "Article on: https://solclover.com", "Article on: https://solclover.com", "Article on: https://solclover.com", " ", "Article on: https://solclover.com", "Article on: https://solclover.com", "Article on: https://solclover.com", "Article on: https://solclover.com", "Article on: https://solclover.com", "Article on: https://solclover.com", "Article on: https://solclover.com", "Article on: https://solclover.com", "Article on: https://solclover.com", " ", "Article on: https://solclover.com", " ", " ", "Article on: https://solclover.com", "Article on: https://solclover.com", "Article on: https://solclover.com", "Article on: https://solclover.com", " ", "Article on: https://solclover.com", " ", "Article on: https://solclover.com", "Article on: https://solclover.com", "Article on: https://solclover.com", "Article on: https://solclover.com", "Article on: https://solclover.com", "Article on: https://solclover.com", " ", " ", "Article on: https://solclover.com", "Article on: https://solclover.com", " ", "Article on: https://solclover.com", "Article on: https://solclover.com", "Article on: https://solclover.com", " ", "Article on: https://solclover.com", "Article on: https://solclover.com", " ", "Article on: https://solclover.com", "Article on: https://solclover.com", "Article on: https://solclover.com", "Article on: https://solclover.com", " ", "Article on: https://solclover.com", "Article on: https://solclover.com", "Article on: https://solclover.com", "Coming Soon", "Coming Soon", " ", "Article on: https://solclover.com", "Article on: https://solclover.com", "Article on: https://solclover.com", " ", "Coming Soon", "Coming Soon", "Coming Soon", "Coming Soon", "Coming Soon", "Coming Soon", "Coming Soon", " ", " ", "Article on: https://solclover.com"], 
  "hovertemplate": "<b>%{label}</b> <br>%{customdata}<br><extra></extra>", 
  "insidetextorientation": "radial"
}
data = Data([trace1])
layout = {
  "width": 900, 
  "height": 900, 
  "margin": {
    "b": 0, 
    "l": 0, 
    "r": 0, 
    "t": 0
  }, 
  "template": {
    "data": {
      "bar": [
        {
          "type": "bar", 
          "marker": {"line": {
              "color": "#E5ECF6", 
              "width": 0.5
            }}, 
          "error_x": {"color": "#2a3f5f"}, 
          "error_y": {"color": "#2a3f5f"}
        }
      ], 
      "pie": [
        {
          "type": "pie", 
          "automargin": True
        }
      ], 
      "table": [
        {
          "type": "table", 
          "cells": {
            "fill": {"color": "#EBF0F8"}, 
            "line": {"color": "white"}
          }, 
          "header": {
            "fill": {"color": "#C8D4E3"}, 
            "line": {"color": "white"}
          }
        }
      ], 
      "carpet": [
        {
          "type": "carpet", 
          "aaxis": {
            "gridcolor": "white", 
            "linecolor": "white", 
            "endlinecolor": "#2a3f5f", 
            "minorgridcolor": "white", 
            "startlinecolor": "#2a3f5f"
          }, 
          "baxis": {
            "gridcolor": "white", 
            "linecolor": "white", 
            "endlinecolor": "#2a3f5f", 
            "minorgridcolor": "white", 
            "startlinecolor": "#2a3f5f"
          }
        }
      ], 
      "mesh3d": [
        {
          "type": "mesh3d", 
          "colorbar": {
            "ticks": "", 
            "outlinewidth": 0
          }
        }
      ], 
      "contour": [
        {
          "type": "contour", 
          "colorbar": {
            "ticks": "", 
            "outlinewidth": 0
          }, 
          "colorscale": [
            [0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1, "#f0f921]
        }
      ], 
      "heatmap": [
        {
          "type": "heatmap", 
          "colorbar": {
            "ticks": "", 
            "outlinewidth": 0
          }, 
          "colorscale": [
            [0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1, "#f0f921]
        }
      ], 
      "scatter": [
        {
          "type": "scatter", 
          "marker": {"colorbar": {
              "ticks": "", 
              "outlinewidth": 0
            }}
        }
      ], 
      "surface": [
        {
          "type": "surface", 
          "colorbar": {
            "ticks": "", 
            "outlinewidth": 0
          }, 
          "colorscale": [
            [0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1, "#f0f921]
        }
      ], 
      "barpolar": [
        {
          "type": "barpolar", 
          "marker": {"line": {
              "color": "#E5ECF6", 
              "width": 0.5
            }}
        }
      ], 
      "heatmapgl": [
        {
          "type": "heatmapgl", 
          "colorbar": {
            "ticks": "", 
            "outlinewidth": 0
          }, 
          "colorscale": [
            [0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1, "#f0f921]
        }
      ], 
      "histogram": [
        {
          "type": "histogram", 
          "marker": {"colorbar": {
              "ticks": "", 
              "outlinewidth": 0
            }}
        }
      ], 
      "parcoords": [
        {
          "line": {"colorbar": {
              "ticks": "", 
              "outlinewidth": 0
            }}, 
          "type": "parcoords"
        }
      ], 
      "scatter3d": [
        {
          "line": {"colorbar": {
              "ticks": "", 
              "outlinewidth": 0
            }}, 
          "type": "scatter3d", 
          "marker": {"colorbar": {
              "ticks": "", 
              "outlinewidth": 0
            }}
        }
      ], 
      "scattergl": [
        {
          "type": "scattergl", 
          "marker": {"colorbar": {
              "ticks": "", 
              "outlinewidth": 0
            }}
        }
      ], 
      "choropleth": [
        {
          "type": "choropleth", 
          "colorbar": {
            "ticks": "", 
            "outlinewidth": 0
          }
        }
      ], 
      "scattergeo": [
        {
          "type": "scattergeo", 
          "marker": {"colorbar": {
              "ticks": "", 
              "outlinewidth": 0
            }}
        }
      ], 
      "histogram2d": [
        {
          "type": "histogram2d", 
          "colorbar": {
            "ticks": "", 
            "outlinewidth": 0
          }, 
          "colorscale": [
            [0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1, "#f0f921]
        }
      ], 
      "scatterpolar": [
        {
          "type": "scatterpolar", 
          "marker": {"colorbar": {
              "ticks": "", 
              "outlinewidth": 0
            }}
        }
      ], 
      "contourcarpet": [
        {
          "type": "contourcarpet", 
          "colorbar": {
            "ticks": "", 
            "outlinewidth": 0
          }
        }
      ], 
      "scattercarpet": [
        {
          "type": "scattercarpet", 
          "marker": {"colorbar": {
              "ticks": "", 
              "outlinewidth": 0
            }}
        }
      ], 
      "scattermapbox": [
        {
          "type": "scattermapbox", 
          "marker": {"colorbar": {
              "ticks": "", 
              "outlinewidth": 0
            }}
        }
      ], 
      "scatterpolargl": [
        {
          "type": "scatterpolargl", 
          "marker": {"colorbar": {
              "ticks": "", 
              "outlinewidth": 0
            }}
        }
      ], 
      "scatterternary": [
        {
          "type": "scatterternary", 
          "marker": {"colorbar": {
              "ticks": "", 
              "outlinewidth": 0
            }}
        }
      ], 
      "histogram2dcontour": [
        {
          "type": "histogram2dcontour", 
          "colorbar": {
            "ticks": "", 
            "outlinewidth": 0
          }, 
          "colorscale": [
            [0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1, "#f0f921]
        }
      ]
    }, 
    "layout": {
      "geo": {
        "bgcolor": "white", 
        "showland": True, 
        "lakecolor": "white", 
        "landcolor": "#E5ECF6", 
        "showlakes": True, 
        "subunitcolor": "white"
      }, 
      "font": {"color": "#2a3f5f"}, 
      "polar": {
        "bgcolor": "#E5ECF6", 
        "radialaxis": {
          "ticks": "", 
          "gridcolor": "white", 
          "linecolor": "white"
        }, 
        "angularaxis": {
          "ticks": "", 
          "gridcolor": "white", 
          "linecolor": "white"
        }
      }, 
      "scene": {
        "xaxis": {
          "ticks": "", 
          "gridcolor": "white", 
          "gridwidth": 2, 
          "linecolor": "white", 
          "zerolinecolor": "white", 
          "showbackground": True, 
          "backgroundcolor": "#E5ECF6"
        }, 
        "yaxis": {
          "ticks": "", 
          "gridcolor": "white", 
          "gridwidth": 2, 
          "linecolor": "white", 
          "zerolinecolor": "white", 
          "showbackground": True, 
          "backgroundcolor": "#E5ECF6"
        }, 
        "zaxis": {
          "ticks": "", 
          "gridcolor": "white", 
          "gridwidth": 2, 
          "linecolor": "white", 
          "zerolinecolor": "white", 
          "showbackground": True, 
          "backgroundcolor": "#E5ECF6"
        }
      }, 
      "title": {"x": 0.05}, 
      "xaxis": {
        "ticks": "", 
        "title": {"standoff": 15}, 
        "gridcolor": "white", 
        "linecolor": "white", 
        "automargin": True, 
        "zerolinecolor": "white", 
        "zerolinewidth": 2
      }, 
      "yaxis": {
        "ticks": "", 
        "title": {"standoff": 15}, 
        "gridcolor": "white", 
        "linecolor": "white", 
        "automargin": True, 
        "zerolinecolor": "white", 
        "zerolinewidth": 2
      }, 
      "mapbox": {"style": "light"}, 
      "ternary": {
        "aaxis": {
          "ticks": "", 
          "gridcolor": "white", 
          "linecolor": "white"
        }, 
        "baxis": {
          "ticks": "", 
          "gridcolor": "white", 
          "linecolor": "white"
        }, 
        "caxis": {
          "ticks": "", 
          "gridcolor": "white", 
          "linecolor": "white"
        }, 
        "bgcolor": "#E5ECF6"
      }, 
      "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], 
      "coloraxis": {"colorbar": {
          "ticks": "", 
          "outlinewidth": 0
        }}, 
      "hovermode": "closest", 
      "colorscale": {
        "diverging": [
          [0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419], 
        "sequential": [
          [0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1, "#f0f921], 
        "sequentialminus": [
          [0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1, "#f0f921]
      }, 
      "hoverlabel": {"align": "left"}, 
      "plot_bgcolor": "#E5ECF6", 
      "paper_bgcolor": "white", 
      "shapedefaults": {"line": {"color": "#2a3f5f"}}, 
      "autotypenumbers": "strict", 
      "annotationdefaults": {
        "arrowhead": 0, 
        "arrowcolor": "#2a3f5f", 
        "arrowwidth": 1
      }
    }
  }, 
  "hovermode": "closest"
}
fig = Figure(data=data, layout=layout)
plot_url = py.plot(fig)