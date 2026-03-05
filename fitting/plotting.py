import matplotlib.pyplot as plt
from corner import corner

def plot_chains(chain):

    nit, _, nparams = chain.shape
    
    fig, axes = plt.subplots(nparams,1, sharex=True)
    for i in range(nparams):
        axes[i].plot(chain[:,:,i], 'k', alpha=0.3)
        axes[i].set_xlim(0.0, nit)
        axes[i].set_ylabel(f'$a_{i}$')
        axes[i].yaxis.set_label_coords(-0.1, 0.5)
    
    axes[-1].set_xlabel('Step Number')
    plt.show()
    return fig

def plot_corner(sampler, discard=100, fig=None, color=None):

    flat_samples = sampler.get_chain(discard=discard, flat=True)
    if fig==None:
        fig = corner(flat_samples, color=color);
        return fig
    else:
        corner(flat_samples, fig=fig, color=color);

def plot_pos(pos):

    nwalkers, nparams = pos.shape

    fig, axes = plt.subplots(1, nparams, figsize=(2*nparams, 2))
    for i in range(nparams):
        axes[i].hist(pos.T[i], bins=nwalkers//10)
        axes[i].set_xlabel(f'$a_{i}$')

    return fig