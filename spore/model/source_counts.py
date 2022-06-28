"""
A module for defining source count models
"""
from cached_property import cached_property
import numpy as np
from scipy.integrate import simps
from spore.common import ensure_unit, nfold_outer
from spore.common.unit_conversions import un
from ._framework import Component as Cmpt

class SourceCounts(Cmpt):
    """
    A general source counts class.

    Parameters
    ----------
    Smax0 : float
        The maximum flux-density of a sample, defined at nu0.

    Smin0 : float, optional
        The minimum flux-density of a sample, defined at nu0

    nu : array-like, optional
        The frequency of the observation, as a ratio to nu0.

    spectral_index : float, optional
        The spectral index of the sample. Note that we assume that all sources have uniform spectral index.

    parameters :
        Any other parameters of the particular model.

    Assumptions
    -----------
    SED's are the same for every object, and are a power law in region of interest.
    """
    _defaults = {}
    def __init__(self,Smax0,Smin0=0,f0=1,spectral_index = 0.8, **parameters):

        # Ensure units
        Smax0 = ensure_unit(Smax0, un.Jy)
        Smin0 = ensure_unit(Smin0, un.Jy)

        self.Smax0 = Smax0
        self.Smin0 = Smin0
        self.f0 = np.atleast_1d(f0)
        self.spectral_index = spectral_index

        self.params = parameters

    def dnds(self, s):
        """
        The source count function
        """
        pass

    @cached_property
    def _svec(self):
        return np.logspace(np.log10(self.Smin0), np.log10(self.Smax0), 1000)

    @cached_property
    def total_number_density(self):
        return simps(self.dnds(self._svec), self._svec)

    @cached_property
    def total_flux_density(self):
        return self.f0 ** -self.spectral_index * simps(self._svec * self.dnds(self._svec), self._svec)

    @cached_property
    def total_squared_flux_density(self):
        return np.outer(self.f0, self.f0) ** (-self.spectral_index) * simps(self._svec ** 2 * self.dnds(self._svec), self._svec)

    @cached_property
    def mean_flux_density_of_source(self):
        return self.total_flux_density/self.total_number_density

    def sample_source_counts(self,N):
        pass

class PowerLawSourceCounts(SourceCounts):

    _defaults = {"alpha":6998.,
                 "beta":1.54}

    def __init__(self,*args,**kwargs):
        super(PowerLawSourceCounts,self).__init__(*args,**kwargs)

        self.params['alpha'] = ensure_unit(self.params['alpha'],1/un.Jy/un.steradian)

    def dnds(self,s):
        s = ensure_unit(s,un.Jy)
        return self.params['alpha']*(s/un.Jy)**-self.params['beta']

    def mu(self,n):
        beta = self.params['beta']
        nufac = nfold_outer(self.f0, n) ** - self.spectral_index
        smx = self.Smax0/un.Jy
        smn = self.Smin0/un.Jy
        return nufac * self.params['alpha'] * (self.Smax0**(n+1) * smx**-beta - self.Smin0**(n+1)*smn**-beta)/(n+1-beta)

    @cached_property
    def total_number_density(self):
        # beta = self.params['beta']
        # return self.params['alpha']*((self.Smax0/un.Jy)**(1-beta) - (self.Smin0/un.Jy)**(1-beta))/(1-beta)
        return self.mu(0)

    @cached_property
    def total_flux_density(self):
        # beta = self.params['beta']
        # return self.nu ** -self.spectral_index * self.params['alpha']*((self.Smax0/un.Jy) ** (2 - beta) - (self.Smin0/un.Jy) ** (2 - beta))/(2. - beta)
        return self.mu(1)

    @cached_property
    def total_squared_flux_density(self):
        # beta = self.params['beta']
        # return np.outer(self.nu,self.nu) ** (-self.spectral_index)*self.params['alpha'] * ((self.Smax0/un.Jy) ** (3 - beta) - (self.Smin0/un.Jy) ** (3 - beta))/(3. - beta)
        return self.mu(2)

    def sample_source_counts(self,N,ret_nu_array=False):
        """
        Generate a sample of N source flux densities, at nu.

        Parameters
        ----------
        N

        Returns
        -------

        """
        beta = self.params['beta']
        smx = (self.Smax0/un.Jy) ** (1 - beta)
        smn = (self.Smin0/un.Jy) ** (1 - beta)
        nu0_sample =((smx - smn)*np.random.uniform(size=N) + smn) ** (1./(1 - beta))

        if ret_nu_array:
            return np.outer(self.f0**-self.spectral_index, nu0_sample * un.Jy)
        else:
            return nu0_sample * un.Jy


class MultiPowerLawSourceCounts(SourceCounts):
    _defaults = {"alpha":6998.,
                 "beta":[1.54],
                 "Sbreak":[0.006]}
    """
    A source count model consisting of continuous (but not continuous derivatives) broken power-laws.
    
    Parameters
    ----------
    Sbreak : list
        The break positions (at nu0) in the counts. Smax and Smin are not included here.
        
    beta : list
        The slopes in each break, going from highest to lowest flux density. (Breaks correspond to upper limit on each
        beta region).
        
    alpha : float
        The normalisation of the counts *in the highest flux density region*.
    """

    def __init__(self,*args,**kwargs):
        super(MultiPowerLawSourceCounts,self).__init__(*args,**kwargs)

        self.params['alpha'] = ensure_unit(self.params['alpha'], 1./un.Jy/un.steradian)

        a = self._get_alpha()
        if np.any(np.array(self.params['Sbreak'])>self.Smax0.value):
            ind = np.where(self.params['Sbreak']>self.Smax0.value)[0][0]
            self.params["Sbreak"] = np.array(self.params["Sbreak"][(ind+1):])
            self.params["alpha"] = a[ind+1]
            self.params["beta"] = np.array(self.params["beta"][(ind+1):])

        if np.any(np.array(self.params['Sbreak'])<self.Smin0.value):
            raise ValueError("All values of Sbreak must be larger than Smin0")

    @property
    def sbreak(self):
        return np.concatenate(([self.Smax0.value], self.params['Sbreak'] , [self.Smin0.value])) * self.Smin0.unit

    def _get_alpha(self):
        """
        The full list of alpha
        """
        alpha = [0]*len(self.params['beta'])
        alpha[0] = self.params['alpha']

        for i,(sb,b) in enumerate(zip(self.sbreak[:-1],self.params['beta'])):

            if i>0:
                alpha[i] = alpha[i-1] * (sb/un.Jy)**(b-self.params['beta'][i-1])
        return alpha

    @cached_property
    def alpha(self):
        return self._get_alpha()

    def dnds(self,s):
        s = ensure_unit(np.atleast_1d(s), un.Jy)
        dn = np.zeros(len(s)) * self.alpha[0].unit
        for i,(sb,b) in enumerate(zip(self.sbreak[:-1],self.params['beta'])):
            mask = np.logical_and(s < sb, s >= self.sbreak[i+1])
            dn[mask] = self.alpha[i] * (s[mask]/un.Jy)**-b
        return dn

    def _get_mu_in_sections(self,n):
        beta = self.params['beta']
        nufac = nfold_outer(self.f0, n) ** - self.spectral_index

        intg = np.zeros(len(beta)) * self.alpha[0].unit * self.Smax0.unit**(n+1)
        for i, (sb, b) in enumerate(zip(self.sbreak[:-1], beta)):
            intg[i] = self.alpha[i]*(sb ** (n+1) * (sb/un.Jy)**-b - self.sbreak[i + 1] ** (n+1) * (self.sbreak[i+1]/un.Jy)** - b)/(n+1 - b)

        return intg, nufac


    # def _get_total_num_dens_in_sections(self):
    #     beta = self.params['beta']
    #
    #     intg = np.zeros(len(beta))
    #     for i, (sb, b) in enumerate(zip(self.sbreak[:-1], beta)):
    #         intg[i] = self.alpha[i] * (sb ** (1 - b) - self.sbreak[i + 1] ** (1 - b)) / (1 - b)
    #     return intg

    @cached_property
    def total_number_density(self):
        intg, nufac = self._get_mu_in_sections(0)
        return np.sum(intg)*nufac #nufac should be 1

    @cached_property
    def total_flux_density(self):
        intg, nufac = self._get_mu_in_sections(1)
        return np.sum(intg)*nufac
        # beta = self.params['beta']
        # intg = 0
        # for i, (sb, b) in enumerate(zip(self.sbreak[:-1], beta)):
        #     intg += self.alpha[i] * (sb**(2-b) - self.sbreak[i+1]**(2-b))/(2-b)

        # return self.nu ** -self.spectral_index * intg

    @cached_property
    def total_squared_flux_density(self):
        intg, nufac = self._get_mu_in_sections(2)
        return np.sum(intg)*nufac

        # beta = self.params['beta']
        # intg = 0
        # for i, (sb, b) in enumerate(zip(self.sbreak[:-1], beta)):
        #     intg += self.alpha[i] * (sb**(3-b) - self.sbreak[i+1]**(3-b))/(3-b)
        #
        # return np.outer(self.nu,self.nu) ** (-self.spectral_index)*intg

    def sample_source_counts(self, N, ret_nu_array=False):
        """
        Generate a sample of source flux densities, over given area, at nu.
        """

        exp_num = self._get_mu_in_sections(0)[0]
        tot_num = np.sum(exp_num)
        exp_frac = exp_num/tot_num

        nsplit = np.unique(np.random.choice(len(exp_frac), size=N, p=exp_frac), return_counts=True)[-1]
        beta = self.params['beta']

        nu0_sample = np.zeros(N)
        nn = 0
        for i, (sb, b,n ) in enumerate(zip(self.sbreak[:-1], beta, nsplit)):
            nu0_sample[nn:nn+n] = (((sb/un.Jy) ** (1 - b) - (self.sbreak[i+1]/un.Jy) ** (1 - b))*np.random.uniform(size=n) + (self.sbreak[i+1]/un.Jy) ** (1 - b)) ** (
                         1./(1 - b))
            nn +=n


        if ret_nu_array:
            return np.outer(self.f0, nu0_sample) * un.Jy
        else:
            return nu0_sample * un.Jy
