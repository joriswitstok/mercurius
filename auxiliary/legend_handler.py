import numpy as np

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as patches

# Tuple and tuple-handler class for two rectangles
class BTuple():
    def __init__(self, btuple, label):
        self.btuple = btuple
        self.label = label
    
    def get_label(self):
        return self.label

class BTupleHandler(matplotlib.legend_handler.HandlerBase):
    def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
        """

        Make a filled-in rectangular handle
        orig_handle.btuple contains four arguments, the last four with a length of either 1 or n_lines:
        - colour
        - alpha
        - linewidth (optional)
        - linestyle (optional)
        - linecolor (optional)
        - linealpha (optional)
        - marker type (optional)
        - marker size (optional)
        - marker edgewidth (optional)
        - marker edgecolor (optional)
        - marker facecolor (optional)
        - marker alpha (optional)

        Example:
        handles = [BTuple((['k'], [0.2], 1.0, '--', 'k', 1.0), label="Model")]
        handles = [BTuple((['k'], [0.2], 1.0, '--', 'k', 1.0, 'o', 20, 1.5, 'b', 'g', 0.7), label="Model")]
        ax.legend(handles=handles, handler_map={BTuple: BTupleHandler())

        """
        
        plot_line = len(orig_handle.btuple) == 6 or len(orig_handle.btuple) == 12
        plot_marker = len(orig_handle.btuple) == 8 or len(orig_handle.btuple) == 12
        
        if plot_marker:
            idx0 = 4 if plot_line else 0
            mtype = orig_handle.btuple[idx0+2]
            msize = orig_handle.btuple[idx0+3]
            mew = orig_handle.btuple[idx0+4]
            mec = orig_handle.btuple[idx0+5]
            mfc = orig_handle.btuple[idx0+6]
            malpha = orig_handle.btuple[idx0+7]
            if not plot_line:
                orig_handle_btuple = orig_handle.btuple
                orig_handle.btuple = orig_handle.btuple[:2]
        else:
            mtype = None
            msize = None
            mew = None
            mec = None
            mfc = None
            malpha = None

        if plot_line:
            linewidth = orig_handle.btuple[2]
            linestyle = orig_handle.btuple[3]
            linecolor = orig_handle.btuple[4]
            linealpha = orig_handle.btuple[5]
            orig_handle_btuple = orig_handle.btuple
            orig_handle.btuple = orig_handle.btuple[:2]
        else:
            linewidth = None
            linestyle = None
            linecolor = None
            linealpha = None
        
        bs = []
        n_bs = len(max(orig_handle.btuple, key=len))

        if n_bs > 0:
            norm_h_edges = np.linspace(0.0, 1.0, n_bs+1)[::-1]
            norm_height = norm_h_edges[0] - norm_h_edges[1]
            assert (norm_h_edges[:-1] - norm_h_edges[1:] - norm_height < 1e-8).all()
        elif n_bs == 0:
            l = plt.Line2D([x0, x0+width], [0.5*height, 0.5*height], color="None")
            bs.append(l)
            return bs
        else:
            raise SystemError("invalid arguments passed to BTuple")
        
        if len(orig_handle.btuple[0]) == 1:
            colors = [orig_handle.btuple[0][0] for il in range(n_bs)]
        else:
            colors = orig_handle.btuple[0]

        if len(orig_handle.btuple[1]) == 1:
            alphas = [orig_handle.btuple[1][0] for il in range(n_bs)]
        else:
            alphas = orig_handle.btuple[1]
        
        for il in range(n_bs):
            b = patches.Rectangle([x0, norm_h_edges[il+1]*height], width, norm_height*height,
                        facecolor=colors[il], edgecolor="None", alpha=alphas[il])
            bs.append(b)

        if plot_line:
            l = plt.Line2D([x0, x0+width], [0.5*height, 0.5*height], linewidth=linewidth, linestyle=linestyle, color=linecolor, alpha=linealpha)
            bs.append(l)

        if plot_marker:
            l = plt.Line2D([x0+0.5*width], [0.5*height], linewidth=0, linestyle="None",
                            marker=mtype, markersize=msize, markeredgewidth=mew, markeredgecolor=mec, markerfacecolor=mfc, alpha=malpha)
            bs.append(l)

        if plot_line or plot_marker:
            # Restore original BTuple
            orig_handle.btuple = orig_handle_btuple

        return bs