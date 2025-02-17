import numpy
import scipy.linalg


class IWasobi():
    def __init__(self, ar_order=10, rmax=0.99, eps0=5.0e-7):
        self.ar_order = ar_order
        self.rmax = rmax
        self.eps0 = eps0
        self.number_iterations = 3
        self.separation_matrix = None
        self.mixing_matrix = None
        self.signal_array_mean = None


    def fit(self, signals: numpy.array):
        signal_array = signals

        channels, samples = signal_array.shape

        signal_array_mean = numpy.mean(signal_array, axis=1)
        signal_array = numpy.transpose(numpy.subtract(numpy.transpose(signal_array), signal_array_mean))
        
        ar_samples = samples - self.ar_order

        correlation_matrix = self.cross_correlation(signal_array, ar_samples, self.ar_order)

        for idx in range(1, self.ar_order + 1):
            idk = channels * idx
            sub_matrix = correlation_matrix[:, idk: idk + channels]
            correlation_matrix[:, idk: idk + channels] = 0.5*(sub_matrix + numpy.transpose(sub_matrix))

        Winit, Ms = self.approximate_joint_diag_uniform(correlation_matrix, 20)

        W = Winit.copy()

        for _ in range(self.number_iterations):
            H = self.weights(Ms, self.rmax, self.eps0)
            W, Ms = self.approximate_joint_diag_weighted(correlation_matrix, H, W, 5)

        sources = numpy.transpose(numpy.add(numpy.transpose(numpy.dot(W, signals)),
                                            numpy.dot(W, signal_array_mean)))

        return (W, signal_array_mean, sources)

    def cross_correlation(self, signal_array: numpy.array, ar_samples: int, ar_order: int) -> numpy.array:
        """
        Correlation matrix
        :param signal_array: multivariate signal, channels first
        :param ar_samples
        :param ar_order
        :param max_iterations: Maximum iterations for the approximation
        """
        number_channels = signal_array.shape[0]
        correlation_matrix = numpy.zeros((number_channels, (ar_order + 1) * number_channels))

        for idx in range(ar_order+1):
            i = number_channels * (idx)
            normalization_factor = 1 / numpy.float64(ar_samples)
            signal = signal_array[:, :ar_samples]
            lagged_signal = numpy.transpose(signal_array[:, idx:ar_samples + idx])
            correlation_matrix[:, i:i + number_channels] = (normalization_factor *
                                                            numpy.dot(signal, lagged_signal))

        return correlation_matrix

    def approximate_joint_diag_uniform(self, matrices: numpy.array, max_iterations: int = 20) -> tuple:
        """
        Approximate joint diagonalization with uniform weights
        :param matrices: matrices to be diagonalized
        :param max_iterations: Maximum iterations for the approximation
        """
        number_channels, number_comparisons = matrices.shape
        comparisons_per_channel = numpy.int32(numpy.floor(number_comparisons/numpy.float64(number_channels)))
        iteration = 0
        epsilon = 1e-7
        improve = 10

        eigenvalues, eigenvectors = numpy.linalg.eig(matrices[:, :number_channels])
        eigenvectors = numpy.real_if_close(eigenvectors, tol=100)
        W_est = numpy.dot(numpy.diag(1.0/numpy.sqrt(eigenvalues)), numpy.transpose(eigenvectors))

        Ms = matrices.copy()
        Rs = numpy.zeros((number_channels, comparisons_per_channel))

        for idx in range(comparisons_per_channel):
            i = number_channels * idx
            vector = matrices[:, i:i + number_channels]
            lagged_vector = numpy.transpose(matrices[:, i:i + number_channels])
            matrices[:, i:i + number_channels] = 0.5 * (vector + lagged_vector)
            Ms[:, i:i + number_channels] = numpy.dot(numpy.dot(W_est, matrices[:, i:i + number_channels]), numpy.transpose(W_est))
            Rs[:, idx] = numpy.diag(Ms[:, i:i + number_channels])

        crit = (Ms**2).sum() - (Rs**2).sum()

        while improve > epsilon and iteration < max_iterations:
            b11, b12, b22, c1, c2 = [], [], [], [], []

            for idx in range(1, number_channels):
                Yim = Ms[0:idx, idx:number_comparisons:number_channels]
                b22.append(numpy.dot(numpy.sum((Rs[idx, :] ** 2), axis=0), numpy.ones((idx, 1))))
                b12.append(numpy.transpose(numpy.dot(Rs[idx, :], numpy.transpose(Rs[:idx, :]))))
                b11.append(numpy.sum((Rs[:idx, :] ** 2), axis=1))
                c2.append(numpy.transpose(numpy.dot(Rs[idx, :], numpy.transpose(Yim))))
                c1.append(numpy.sum((Rs[:idx, :] * Yim), axis=1))

            b22 = numpy.squeeze(numpy.vstack(b22))
            b12 = numpy.hstack(b12)
            b11 = numpy.hstack(b11)
            c2 = numpy.hstack(c2)
            c1 = numpy.hstack(c1)

            det0 = b11*b22-b12**2
            d1 = (c1*b22-b12*c2)/det0
            d2 = (b11*c2-b12*c1)/det0

            m = 0
            A0 = numpy.eye(number_channels)

            for id in range(1, number_channels):
                A0[id, 0:id] = d1[m:m+id]
                A0[0:id, id] = d2[m:m+id]
                m += id

            Ainv = numpy.linalg.inv(A0)
            W_est = numpy.dot(Ainv, W_est)
            Raux = numpy.dot(numpy.dot(W_est, matrices[:, :number_channels]), W_est.T)
            aux = 1/numpy.sqrt(numpy.diag(Raux))
            W_est = numpy.dot(numpy.diag(aux), W_est)

            for k in range(comparisons_per_channel):
                ini = k*number_channels
                Ms[:, ini:ini+number_channels] = numpy.dot(numpy.dot(W_est, matrices[:, ini:ini+number_channels]), W_est.T)
                Rs[:, k] = numpy.diag(Ms[:, ini:ini + number_channels])

            critic = (Ms**2).sum() - (Rs**2).sum()
            improve = numpy.abs(critic - crit)
            crit = critic
            iteration += 1

        return W_est, Ms

    def weights(self, Ms, rmax, eps0) -> numpy.array:
        number_channels, number_comparisons = Ms.shape
        comparisons_per_channel = numpy.int32(numpy.floor(number_comparisons/numpy.float64(number_channels)))
        d2 = numpy.int32(number_channels * (number_channels - 1)/2.0)
        R = numpy.zeros((comparisons_per_channel, number_channels))

        for index in range(comparisons_per_channel):
            id = index * number_channels
            R[index, :] = numpy.diag(Ms[:, id:id+number_channels])

        ARC, sigmy = self.armodel(R, rmax)
        AR3 = numpy.zeros((2*comparisons_per_channel-1, d2))
        ll = 0
        for i in range(1, number_channels):
            for k in range(i):
                AR3[:, ll] = numpy.convolve(ARC[:, i], ARC[:, k])
                ll += 1

        phi = self.ar2r(AR3)
        H = self.THinv5(phi, comparisons_per_channel, d2, eps0*phi[0, :])

        im = 0
        for i in range(1, number_channels):
            for k in range(i):
                fact = 1/(sigmy[i]*sigmy[k])
                imm = im*comparisons_per_channel
                H[:, imm:imm+comparisons_per_channel] = H[:, imm:imm+comparisons_per_channel]*fact
                im += 1

        return H

    def armodel(self, R, rmax):
        """
        Compute AR coefficients of the sources given covariance functions
        but if the zeros have magnitude > rmax, the zeros are pushed back.
        """
        M, d = R.shape
        AR = numpy.zeros((M, d))

        for id in range(d):
            AR[:, id] = numpy.r_[1, numpy.linalg.lstsq(-scipy.linalg.toeplitz(R[:M-1, id], R[:M-1, id].T),
                                 R[1:M, id])[0]]
            v = numpy.roots(AR[:, id])
            vs = 0.5*(numpy.sign(numpy.abs(v)-1)+1)
            v = (1-vs)*v + vs/numpy.conj(v)
            vmax = numpy.max(numpy.abs(v))

            if vmax > rmax:
                v = v*rmax/vmax

            AR[:, id] = numpy.real(numpy.poly(v).T)

        Rs = self.ar2r(AR)
        sigmy = R[0, :]/Rs[0, :]
        return AR, sigmy

    def ar2r(self, a):
        """
        Computes covariance function of AR processes from
        the autoregressive coefficients using an inverse Schur algorithm
        and an inverse Levinson algorithm
        """
        if a.shape[0] == 1:
            a = a.T

        p, m = a.shape
        alfa = a.copy()
        K = numpy.zeros((p, m))
        p -= 1

        for n in range(p)[::-1]:
            K[n, :] = -a[n+1, :]
            for k in range(n):
                alfa[k+1, :] = (a[k+1, :]+K[n, :]*a[n-k, :])/(1-K[n, :]**2)
            a = alfa.copy()

        r = numpy.zeros((p+1, m))
        r[0, :] = 1/numpy.prod(1-K**2, 0)
        f = r.copy()
        b = f.copy()

        for k in range(p):
            for n in range(k+1)[::-1]:
                K_n = K[n, :]
                f[n, :] = f[n+1, :] + K_n*b[k-n, :]
                b[k-n, :] = -K_n*f[n+1, :]+(1-K_n**2)*b[k-n, :]
            b[k+1, :] = f[0, :]
            r[k+1, :] = f[0, :]

        return r

    def THinv5(self, phi, K, M, eps):
        """
        Implements fast (complexity O(M*K^2))
        Ccomputation of the following piece of code:
        """
        C = []
        for im in range(M):
            A = (scipy.linalg.toeplitz(phi[:K, im], phi[:K, im].T) +
                 scipy.linalg.hankel(phi[:K, im], phi[K-1:2*K, im].T) +
                 eps[im]*numpy.eye(K))
            C.append(numpy.linalg.inv(A))

        return numpy.concatenate(C, axis=1)

    def approximate_joint_diag_weighted(self, M, H, W_est0=None, maxnumit=100):
        """
        Approximate joint diagonalization with non-uniform weights
        :param M: matrices to be diagonalized
        :param H: diagonal blocks of the weight matrix
        :param W_est0: initial estimate of the demixing matrix, if available
        :param maxnumit: maximum number of iterations
        :returns W_est: estimated demixing matrix
                        such that W_est * M_k * W_est' are roughly diagonal
        :returns Ms: diagonalized matrices composed of W_est*M_k*W_est'
        """
        d, Md = M.shape
        L = numpy.int32(numpy.floor(Md/numpy.float64(d)))
        dd2 = numpy.int32(d*(d-1)/2.0)
        Md = L*d

        if W_est0 is None:
            E, H = numpy.linalg.eig(M[:, :d])
            H = numpy.real_if_close(H, tol=100)
            W_est = numpy.dot(numpy.diag(1/numpy.sqrt(E)), H.T)
        else:
            W_est = W_est0

        Ms = M.copy()
        Rs = numpy.zeros((d, L))

        for k in range(L):
            ini = k*d
            M[:, ini:ini+d] = 0.5*(M[:, ini:ini+d]+M[:, ini:ini+d].T)
            Ms[:, ini:ini+d] = numpy.dot(numpy.dot(W_est, M[:, ini:ini+d]), W_est.T)
            Rs[:, k] = numpy.diag(Ms[:, ini:ini+d])

        for _ in range(maxnumit):
            b11 = numpy.zeros((dd2, 1))
            b12 = numpy.zeros((dd2, 1))
            b22 = numpy.zeros((dd2, 1))
            c1 = numpy.zeros((dd2, 1))
            c2 = numpy.zeros((dd2, 1))
            m = 0

            for id in range(1, d):
                for id2 in range(id):
                    im = m*L
                    Wm = H[:, im:im+L]
                    Yim = Ms[id, id2:Md:d]
                    Rs_id = Rs[id, :]
                    Rs_id2 = Rs[id2, :]
                    Wlam1 = numpy.dot(Wm, Rs_id.T)
                    Wlam2 = numpy.dot(Wm, Rs_id2.T)
                    b11[m] = numpy.dot(Rs_id2, Wlam2)
                    b12[m] = numpy.dot(Rs_id, Wlam2)
                    b22[m] = numpy.dot(Rs_id, Wlam1)
                    c1[m] = numpy.dot(Wlam2.T, Yim.T)
                    c2[m] = numpy.dot(Wlam1.T, Yim.T)
                    m += 1

            det0 = b11*b22-b12**2
            d1 = (c1*b22-b12*c2)/det0
            d2 = (b11*c2-b12*c1)/det0
            m = 0
            A0 = numpy.eye(d)

            for id in range(1, d):
                A0[id, 0:id] = d1[m:m+id, 0]
                A0[0:id, id] = d2[m:m+id, 0]
                m += id

            Ainv = numpy.linalg.inv(A0)
            W_est = numpy.dot(Ainv, W_est)
            Raux = numpy.dot(numpy.dot(W_est, M[:, :d]), W_est.T)
            aux = 1/numpy.sqrt(numpy.diag(Raux))
            W_est = numpy.dot(numpy.diag(aux), W_est)

            for k in range(L):
                ini = k*d
                Ms[:, ini:ini+d] = numpy.dot(numpy.dot(W_est, M[:, ini:ini+d]), W_est.T)
                Rs[:, k] = numpy.diag(Ms[:, ini:ini+d])

        return W_est, Ms

    def fit_transform(self, signal: numpy.array) -> numpy.array:
        """
        Interface function for iWASOBI algorithm.
        IMPORTANT: If the maximum spread of eigenvalues is violated, the most redundant
        mixtures will be discarded in the estimation process.
        IMPORTANT: If the unkonwn mixing matrix is not of
        full-column rank we will have that size(A,1)>size(W,1).
        Output: Separation matrix & Mixing matrix
        :param signal: array of eeg signal (channels first)
        """
        separation_matrix, signal_array_mean, separated_sources = self.fit(signal)
        mixing_matrix = numpy.linalg.pinv(separation_matrix)

        self.signal_array_mean = signal_array_mean
        self.separation_matrix = separation_matrix
        self.mixing_matrix = mixing_matrix
        return separated_sources

    def inverse_transform(self, sources: numpy.array) -> numpy.array:
        intermediate_mix = numpy.add(numpy.transpose(sources),
                                     (-1) * numpy.dot(self.separation_matrix, self.signal_array_mean))
        return numpy.dot(self.mixing_matrix, numpy.transpose(intermediate_mix))
    
        #sources = numpy.transpose(numpy.add(numpy.transpose(numpy.dot(W, signals)),
        #                                   numpy.dot(W, signal_array_mean)))
        #return numpy.dot(self.mixing_matrix, sources)


def principal_component_analysis(signal: numpy.array, number_components: str = "default",
                                 eigenvalues_ratio: str = 1e8) -> tuple:
    """
    Principal component analysis, returns the PCA matrix and the principal
    components. The number of components is 'number_components' unless the
    ratio between the maximum and minimum covariance eigenvalue is below
    eigenvalues_ratio
    """
    number_components = signal.shape[0] if number_components == "default" else number_components

    covariance_matrix = numpy.cov(signal)
    eigenvalues, eigenvectors = numpy.linalg.eig(covariance_matrix)
    absolute_eigenvalues = numpy.abs(eigenvalues)
    sorted_indices = numpy.argsort(absolute_eigenvalues, )[::-1]
    absolute_eigenvalues = absolute_eigenvalues[sorted_indices]

    while absolute_eigenvalues[0] / absolute_eigenvalues[number_components-1] > eigenvalues_ratio:
        number_components -= 1

    eigenvectors = eigenvectors[:, sorted_indices[:number_components]]
    eigenvalues = numpy.diag(eigenvalues[sorted_indices[:number_components]] ** (-0.5))

    weights = numpy.matmul(eigenvalues, numpy.transpose(eigenvectors))
    principal_components = numpy.matmul(weights, signal)

    return weights, principal_components
