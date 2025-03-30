import numpy
import logging
import scipy.linalg
from typing import Tuple
from . import settings


class IWasobi():

    def __init__(self, autoregressive_order=10, rmax=0.99, epsilon_0=5.0e-7):
        """
        :param autoregressive_order: maximum AR order of the separated sources
        param rmax: maximum magnitude of poles of the AR sources
        :param epsilon_0: constant to control condition number of weight matrices
        """
        self.autoregressive_order = autoregressive_order
        self.rmax = rmax
        self.epsilon_0 = epsilon_0
        self.number_iterations = 3
        self.separation_matrix = None
        self.mixing_matrix = None
        self.signal_array_mean = None

    def fit(self, signals: numpy.array)-> tuple:
        """
        Implementation of Blind Source Separation through WASOBI algorithm
        Compute the sources and separation matrix.
        
        Reference matlab implementation: github.com/compmem/ptsa
        :param signals: matrix with the mixed signals [sources x samples]
        """
        signal_array = signals

        channels, samples = signal_array.shape

        signal_array_mean = numpy.mean(signal_array, axis=1)
        signal_array = numpy.transpose(numpy.subtract(numpy.transpose(signal_array), signal_array_mean))

        autoregressive_samples = samples - self.autoregressive_order

        correlation_matrix = self.cross_correlation(signal_array, autoregressive_samples, self.autoregressive_order)

        for idx in range(1, self.autoregressive_order + 1):
            idk = channels * idx
            sub_matrix = correlation_matrix[:, idk: idk + channels]
            correlation_matrix[:, idk: idk + channels] = 0.5 * (sub_matrix + numpy.transpose(sub_matrix))

        weights_init, diagonalized_matrices = self.approximate_joint_diag_uniform(correlation_matrix, 20)

        weights = weights_init.copy()

        for _ in range(self.number_iterations):
            h = self.weights(diagonalized_matrices, self.rmax, self.epsilon_0)
            weights, diagonalized_matrices = self.approximate_joint_diag_weighted(correlation_matrix, h, weights, 5)

        sources = numpy.transpose(numpy.add(numpy.transpose(numpy.dot(weights, signals)),
                                            numpy.dot(weights, signal_array_mean)))

        return (weights, signal_array_mean, sources)

    def cross_correlation(self, signal_array: numpy.array, ar_samples: int, autoregressive_order: int) -> numpy.array:
        """
        Correlation matrix
        :param signal_array: multivariate signal, channels first
        :param ar_samples
        :param autoregressive_order
        :param max_iterations: Maximum iterations for the approximation
        """
        number_channels = signal_array.shape[0]
        correlation_matrix = numpy.zeros((number_channels, (autoregressive_order + 1) * number_channels))

        for idx in range(autoregressive_order + 1):
            i = number_channels * (idx)
            normalization_factor = 1 / numpy.float64(ar_samples)
            signal = signal_array[:, :ar_samples]
            lagged_signal = numpy.transpose(signal_array[:, idx:ar_samples + idx])
            correlation_matrix[:, i:i + number_channels] = (normalization_factor *
                                                            numpy.dot(signal, lagged_signal))

        return correlation_matrix

    def approximate_joint_diag_uniform(self, matrices: numpy.array, max_iterations: int = 20) -> Tuple[numpy.array, numpy.array]:
        """
        Approximate joint diagonalization with uniform weights
        :param matrices: matrices to be diagonalized
        :param max_iterations: Maximum iterations for the approximation
        """
        number_channels, number_comparisons = matrices.shape
        comparisons_per_channel = numpy.int32(numpy.floor(number_comparisons / numpy.float64(number_channels)))
        iteration = 0
        epsilon = 1e-7
        improvement = 10

        eigenvalues, eigenvectors = numpy.linalg.eig(matrices[:, :number_channels])
        eigenvectors = numpy.real_if_close(eigenvectors, tol=100)
        weights_estimation = numpy.dot(numpy.diag(1.0 / numpy.sqrt(eigenvalues)), numpy.transpose(eigenvectors))

        diagonalized_matrices = matrices.copy()
        lagged_cov_matrix = numpy.zeros((number_channels, comparisons_per_channel))

        for idx in range(comparisons_per_channel):
            i = number_channels * idx
            vector = matrices[:, i:i + number_channels]
            lagged_vector = numpy.transpose(matrices[:, i:i + number_channels])
            matrices[:, i:i + number_channels] = 0.5 * (vector + lagged_vector)
            diagonalized_matrices[:, i:i + number_channels] = numpy.dot(numpy.dot(weights_estimation, matrices[:, i:i + number_channels]),
                                                     numpy.transpose(weights_estimation))
            lagged_cov_matrix[:, idx] = numpy.diag(diagonalized_matrices[:, i:i + number_channels])

        crit = (diagonalized_matrices**2).sum() - (diagonalized_matrices**2).sum()

        while improvement > epsilon and iteration < max_iterations:
            b11, b12, b22, c1, c2 = [], [], [], [], []

            for idx in range(1, number_channels):
                yim = diagonalized_matrices[0:idx, idx:number_comparisons:number_channels]
                b22.append(numpy.dot(numpy.sum((lagged_cov_matrix[idx, :] ** 2), axis=0), numpy.ones((idx, 1))))
                b12.append(numpy.transpose(numpy.dot(lagged_cov_matrix[idx, :], numpy.transpose(lagged_cov_matrix[:idx, :]))))
                b11.append(numpy.sum((lagged_cov_matrix[:idx, :] ** 2), axis=1))
                c2.append(numpy.transpose(numpy.dot(lagged_cov_matrix[idx, :], numpy.transpose(yim))))
                c1.append(numpy.sum((lagged_cov_matrix[:idx, :] * yim), axis=1))

            b22 = numpy.squeeze(numpy.vstack(b22))
            b12 = numpy.hstack(b12)
            b11 = numpy.hstack(b11)
            c2 = numpy.hstack(c2)
            c1 = numpy.hstack(c1)

            determinant_0 = b11 * b22 - b12 ** 2
            d1 = (c1 * b22 - b12 * c2) / determinant_0
            d2 = (b11 * c2 - b12 * c1) / determinant_0

            m = 0
            a0 = numpy.eye(number_channels)

            for id in range(1, number_channels):
                a0[id, 0:id] = d1[m:m + id]
                a0[0:id, id] = d2[m:m + id]
                m += id

            a_inv = numpy.linalg.inv(a0)
            weights_estimation = numpy.dot(a_inv, weights_estimation)
            r_aux = numpy.dot(numpy.dot(weights_estimation, matrices[:, :number_channels]), weights_estimation.T)
            aux = 1 / numpy.sqrt(numpy.diag(r_aux))
            weights_estimation = numpy.dot(numpy.diag(aux), weights_estimation)

            for k in range(comparisons_per_channel):
                ini = k * number_channels
                diagonalized_matrices[:, ini:ini + number_channels] = numpy.dot(numpy.dot(weights_estimation, matrices[:, ini:ini + number_channels]), weights_estimation.T)
                lagged_cov_matrix[:, k] = numpy.diag(diagonalized_matrices[:, ini:ini + number_channels])

            critic = (diagonalized_matrices**2).sum() - (lagged_cov_matrix**2).sum()
            improvement = numpy.abs(critic - crit)
            crit = critic
            iteration += 1

        return weights_estimation, diagonalized_matrices

    def weights(self, diagonalized_matrices, rmax, epsilon_0) -> numpy.array:
        number_channels, number_comparisons = diagonalized_matrices.shape
        comparisons_per_channel = numpy.int32(numpy.floor(number_comparisons / numpy.float64(number_channels)))
        d2 = numpy.int32(number_channels * (number_channels - 1) / 2.0)
        r = numpy.zeros((comparisons_per_channel, number_channels))

        for index in range(comparisons_per_channel):
            id = index * number_channels
            r[index, :] = numpy.diag(diagonalized_matrices[:, id:id + number_channels])

        arc, sigmy = self.armodel(r, rmax)

        ar3 = numpy.zeros((2 * comparisons_per_channel - 1, d2))
        ll = 0
        for i in range(1, number_channels):
            for k in range(i):
                ar3[:, ll] = numpy.convolve(arc[:, i], arc[:, k])
                ll += 1

        phi = self.ar2r(ar3)
        h = self.th_inv5(phi, comparisons_per_channel, d2, epsilon_0 * phi[0, :])

        im = 0
        for i in range(1, number_channels):
            for k in range(i):
                fact = 1 / (sigmy[i] * sigmy[k])
                imm = im * comparisons_per_channel
                h[:, imm:imm + comparisons_per_channel] = h[:, imm:imm + comparisons_per_channel] * fact
                im += 1

        return h

    def armodel(self, r, rmax) -> Tuple[numpy.array, numpy.array]:
        """
        Compute AR coefficients of the sources given covariance functions
        but if the zeros have magnitude > rmax, the zeros are pushed back.
        """
        m, d = r.shape
        ar = numpy.zeros((m, d))

        for id in range(d):
            ar[:, id] = numpy.r_[1, numpy.linalg.lstsq(-scipy.linalg.toeplitz(r[:m - 1, id], r[:m - 1, id].T),
                                                       r[1:m, id])[0]]
            v = numpy.roots(ar[:, id])
            vs = 0.5 * (numpy.sign(numpy.abs(v) - 1) + 1)
            v = (1 - vs) * v + vs / numpy.conj(v)
            vmax = numpy.max(numpy.abs(v))

            if vmax > rmax:
                v = v * rmax / vmax

            ar[:, id] = numpy.real(numpy.poly(v).T)

        lagged_cov_matrix = self.ar2r(ar)
        sigmy = r[0, :] / lagged_cov_matrix[0, :]
        return ar, sigmy

    def ar2r(self, a) -> numpy.array:
        """
        Computes covariance function of AR processes from
        the autoregressive coefficients using an inverse Schur algorithm
        and an inverse Levinson algorithm
        """
        if a.shape[0] == 1:
            a = a.T

        p, m = a.shape
        alfa = a.copy()
        k_matrix = numpy.zeros((p, m))
        p -= 1

        for n in range(p)[::-1]:
            k_matrix[n, :] = -a[n + 1, :]
            for k in range(n):
                alfa[k + 1, :] = (a[k + 1, :] + k_matrix[n, :] * a[n - k, :]) / (1 - k_matrix[n, :] ** 2)
            a = alfa.copy()

        r = numpy.zeros((p + 1, m))
        r[0, :] = 1 / numpy.prod(1 - k_matrix ** 2, 0)
        f = r.copy()
        b = f.copy()

        for k in range(p):
            for n in range(k + 1)[::-1]:
                k_n = k_matrix[n, :]
                f[n, :] = f[n + 1, :] + k_n * b[k - n, :]
                b[k - n, :] = -k_n * f[n + 1, :] + (1 - k_n ** 2) * b[k - n, :]
            b[k + 1, :] = f[0, :]
            r[k + 1, :] = f[0, :]

        return r

    def th_inv5(self, phi, k_matrix, m, eps):
        """
        """
        c = []
        for im in range(m):
            a = (scipy.linalg.toeplitz(phi[:k_matrix, im], phi[:k_matrix, im].T) +
                 scipy.linalg.hankel(phi[:k_matrix, im], phi[k_matrix - 1:2 * k_matrix, im].T) +
                 eps[im] * numpy.eye(k_matrix))
            c.append(numpy.linalg.inv(a))

        return numpy.concatenate(c, axis=1)

    def approximate_joint_diag_weighted(self, m_matrix, h, weights_estimation_0=None, maxnumit=100) -> Tuple[numpy.array, numpy.array]:
        """
        Approximate joint diagonalization with non-uniform weights
        :param m: matrices to be diagonalized
        :param h: diagonal blocks of the weight matrix
        :param weights_estimation_0: initial estimate of the demixing matrix, if available
        :param maxnumit: maximum number of iterations
        :returns weights_estimation: estimated demixing matrix
                                     such that weights_estimation * M_k * weights_estimation' are roughly diagonal
        :returns diagonalized_matrices: diagonalized matrices composed of weights_estimation * m_k * weights_estimation'
        """
        d, md = m_matrix.shape
        l_vector = numpy.int32(numpy.floor(md / numpy.float64(d)))
        dd2 = numpy.int32(d * (d - 1) / 2.0)
        md = l_vector * d

        if weights_estimation_0 is None:
            e, h = numpy.linalg.eig(m_matrix[:, :d])
            h = numpy.real_if_close(h, tol=100)
            weights_estimation = numpy.dot(numpy.diag(1 / numpy.sqrt(e)), h.T)
        else:
            weights_estimation = weights_estimation_0

        diagonalized_matrices = m_matrix.copy()
        lagged_cov_matrix = numpy.zeros((d, l_vector))

        for k in range(l_vector):
            ini = k * d
            m_matrix[:, ini:ini + d] = 0.5 * (m_matrix[:, ini:ini + d] + m_matrix[:, ini:ini + d].T)
            diagonalized_matrices[:, ini:ini + d] = numpy.dot(numpy.dot(weights_estimation, m_matrix[:, ini:ini + d]), weights_estimation.T)
            lagged_cov_matrix[:, k] = numpy.diag(diagonalized_matrices[:, ini:ini + d])

        for _ in range(maxnumit):
            b11 = numpy.zeros((dd2, 1))
            b12 = numpy.zeros((dd2, 1))
            b22 = numpy.zeros((dd2, 1))
            c1 = numpy.zeros((dd2, 1))
            c2 = numpy.zeros((dd2, 1))
            m = 0

            for id in range(1, d):
                for id2 in range(id):
                    im = m * l_vector
                    wm = h[:, im:im + l_vector]
                    yim = diagonalized_matrices[id, id2:md:d]
                    rs_id = lagged_cov_matrix[id, :]
                    rs_id2 = lagged_cov_matrix[id2, :]
                    wlam1 = numpy.dot(wm, rs_id.T)
                    wlam2 = numpy.dot(wm, rs_id2.T)
                    b11[m] = numpy.dot(rs_id2, wlam2)
                    b12[m] = numpy.dot(rs_id, wlam2)
                    b22[m] = numpy.dot(rs_id, wlam1)
                    c1[m] = numpy.dot(wlam2.T, yim.T)
                    c2[m] = numpy.dot(wlam1.T, yim.T)
                    m += 1

            determinant_0 = b11 * b22 - b12 ** 2
            d1 = (c1 * b22 - b12 * c2) / determinant_0
            d2 = (b11 * c2 - b12 * c1) / determinant_0
            m = 0
            a0 = numpy.eye(d)

            for id in range(1, d):
                a0[id, 0:id] = d1[m:m + id, 0]
                a0[0:id, id] = d2[m:m + id, 0]
                m += id

            a_inv = numpy.linalg.inv(a0)
            weights_estimation = numpy.dot(a_inv, weights_estimation)
            r_aux = numpy.dot(numpy.dot(weights_estimation, m_matrix[:, :d]), weights_estimation.T)
            aux = 1 / numpy.sqrt(numpy.diag(r_aux))
            weights_estimation = numpy.dot(numpy.diag(aux), weights_estimation)

            for k in range(l_vector):
                ini = k * d
                diagonalized_matrices[:, ini:ini + d] = numpy.dot(numpy.dot(weights_estimation, m_matrix[:, ini:ini + d]), weights_estimation.T)
                lagged_cov_matrix[:, k] = numpy.diag(diagonalized_matrices[:, ini:ini + d])

        return weights_estimation, diagonalized_matrices

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


class CanonicalCorrelation():

    def __init__(self, delay: int = 1):
        self.delay = delay

    def fit(self, signals: numpy.array) -> Tuple[numpy.array, numpy.array]:
        """
        Implementation of Blind Source Separation through Canonical Correlation Analysis
        Compute the canonical components and separation matrix.
        
        Reference matlab implementation: github.com/germangh/eeglab_plugin_aar
        :param signals: matrix with the mixed signals [sources x samples]
        """
        _, samples = tuple(signals.shape)

        y_matrix = signals[:, self.delay:]
        x_matrix = signals[:, : -self.delay]

        correlation_yy = numpy.matmul(y_matrix, numpy.transpose(y_matrix)) * (1 / samples)
        correlation_xx = numpy.matmul(x_matrix, numpy.transpose(x_matrix)) * (1 / samples)

        correlation_xy = numpy.matmul(x_matrix, numpy.transpose(y_matrix)) * (1 / samples)
        correlation_yx = numpy.transpose(correlation_xy)

        inv_correlation_yy = numpy.linalg.pinv(correlation_yy)
        inv_correlation_xx = numpy.linalg.pinv(correlation_xx)

        intermediate_matrix = numpy.matmul(inv_correlation_xx, correlation_xy)
        intermediate_matrix = numpy.matmul(intermediate_matrix, inv_correlation_yy)
        intermediate_matrix = numpy.matmul(intermediate_matrix, correlation_yx)
        eigenvalues, eigenvectors = numpy.linalg.eig(intermediate_matrix)

        sorted_indices = numpy.argsort(numpy.sqrt(abs(eigenvalues)))
        eigenvectors = eigenvectors[:, sorted_indices]

        sources = numpy.dot(eigenvectors, signals)

        return eigenvectors, sources

    def fit_transform(self, signal: numpy.array) -> numpy.array:
        """
        Interface function for BSSCA algorithm.
        IMPORTANT: If the maximum spread of eigenvalues is violated, the most redundant
        mixtures will be discarded in the estimation process.
        IMPORTANT: If the unknown mixing matrix is not of
        full-column rank we will have that size(A,1) > size(W,1).
        :param signal: array of eeg signal [channels x samples]
        """
        separation_matrix, separated_sources = self.fit(signal)
        mixing_matrix = numpy.linalg.pinv(separation_matrix)
        self.separation_matrix = separation_matrix
        self.mixing_matrix = mixing_matrix
        return separated_sources

    def inverse_transform(self, sources: numpy.array) -> numpy.array:
        """
        Return the mixed sources
        :param sources: matrix with the eeg sources [sources x samples]
        """
        return numpy.dot(self.mixing_matrix, sources)


class EogDenoiser():

    def __init__(self, sampling_frequency: int):
        """
        :param sampling_frequency: sampling_frequency [Hz]
        """
        self.sampling_frequency = sampling_frequency
        self.segment_lenght = settings["eog_denoiser_segment_lenght"]
        self.sources_tol = settings["eog_denoiser_sources_tol"]

    def __str__(self):
        for idx, dimension in enumerate(self.fractal_dimensions):
            print(f"Source = {idx + 1}, fd = {dimension}")
        return "Done!"

    def fit_iwasobi(self, segment: numpy.array) -> numpy.array:
        """
        Estimate the sources for a single eeg segment
        :param segment: matrix with the eeg sources [sources x samples]
        """
        self.iwasobi = IWasobi()
        sources = self.iwasobi.fit_transform(segment)
        return sources

    def fit_fractal_dimensions(self, sources: numpy.array) -> None:
        """
        Estimate fractal dimension for each source
        :param sources: matrix with the eeg sources [sources x samples]
        """
        fractal_dimensions = []
        for idx in range(sources.shape[0]):
            fractal_dimemsion = self.sevcik_fractal_dimension(sources[idx, :])
            fractal_dimensions.append(fractal_dimemsion)
        sources = sources
        self.fractal_dimensions = numpy.array(fractal_dimensions)

    def sevcik_fractal_dimension(self, source: numpy.array) -> float:
        """
        Computes the Sevcik dimension
        :param source: 1d-source
        """
        x_axis_span = len(source)
        y_axis_span = max(source) - min(source)
        y_normalized_axis = source - max(source) / y_axis_span
        x_normalized_axis = (1 / (x_axis_span - 1)) * numpy.ones((1, x_axis_span - 1))
        y_axis_differences = y_normalized_axis[1:] - y_normalized_axis[: -1]
        waveform_lenght = numpy.sum((numpy.sqrt(x_normalized_axis ** 2 + y_axis_differences ** 2)))
        dimension = 1 + numpy.log(waveform_lenght / numpy.log(2 * (x_axis_span - 1)))
        return dimension

    def remove_low_dimension_sources(self, sources: numpy.array) -> numpy.array:
        """
        Remove EOG components according to their fractal dimensions
        :param sources: matrix with the eeg sources [sources x samples]
        """
        sorted_indices = numpy.argsort(self.fractal_dimensions)
        self.fractal_dimensions = self.fractal_dimensions[sorted_indices]
        fractal_distances = self.fractal_dimensions[1:] - self.fractal_dimensions[0:-1]
        limit_index = 0

        for idx in range(1, len(fractal_distances) - 1):
            if fractal_distances[idx] < fractal_distances[idx - 1]:
                limit_index = idx
                break

        if limit_index < self.sources_tol[0]:
            limit_index = self.sources_tol[0]
        elif limit_index > self.sources_tol[1]:
            limit_index = self.sources_tol[1]

        logging.info(f"Number of sources to be removed = {limit_index}, indices: {sorted_indices[:limit_index]}")
        for idx in sorted_indices[:limit_index]:
            sources[idx, :] = 0
        return sources

    def apply_by_segments(self, eeg_array: numpy.array) -> numpy.array:
        """
        Remove the EOG sources by splitting the eeg recording
        into smaller windows. Each window is processed independently.
        :param eeg_array: [channels x samples]
        """
        segments = []
        segment_lenght = self.segment_lenght * self.sampling_frequency
        next_idx = segment_lenght

        for previous_idx in range(0, eeg_array.shape[1], segment_lenght):
            segment = eeg_array[:, previous_idx: next_idx]
            segments.append(segment)
            next_idx += segment_lenght

        clean_eegs = []
        aggregated_sources = []
        for segment in segments:
            sources = self.fit_iwasobi(segment)
            aggregated_sources.append(numpy.copy(sources))
            self.fit_fractal_dimensions(sources)
            clean_sources = self.remove_low_dimension_sources(sources)
            clean_eeg = self.iwasobi.inverse_transform(clean_sources)
            clean_eegs.append(clean_eeg)

        return numpy.concat(aggregated_sources, axis=1), numpy.concat(clean_eegs, axis=1)


class EmgDenoiser():

    def __init__(self, sampling_frequency: int = None):
        """
        :param sampling_frequency: sampling_frequency [Hz]
        """
        self.sources = None
        self.segment_lenght = settings["emg_denoiser_segment_lenght"]
        self.frequency_emg = settings["emg_denoiser_emg_freq"]
        self.sources_tol = settings["eog_denoiser_sources_tol"]
        self.ratio = settings["emg_denoiser_ratio"]
        self.sampling_frequency = sampling_frequency

    def __str__(self):
        for idx, dimension in enumerate(self.psd_ratio):
            print(f"Source = {idx + 1}, fd = {dimension}")
        return "Done!"

    def fit_canonical_correlation(self, segment: numpy.array) -> numpy.array:
        """
        Estimate the sources for a single eeg segment
        :param segment: matrix with the eeg sources [sources x samples]
        """
        self.ccanalysis = CanonicalCorrelation()
        sources = self.ccanalysis.fit_transform(segment)
        return sources

    def fit_psd_ratio(self, sources: numpy.array) -> None:
        """
        Compute the PSD ratio between typical eeg and emg bands
        :param source: matrix with the eeg sources [sources x samples]
        """
        self.psd_ratio = []
        _, samples = tuple(sources.shape)
        segment_length = min([2 * self.sampling_frequency, int(samples / 2)])
        nfft_length = 2 ** numpy.ceil(numpy.log2(min([2 * self.sampling_frequency, int(samples / 2)])))

        source_mean = numpy.mean(sources, axis=0)
        centered_sources = numpy.subtract(sources, source_mean)

        for idx in range(centered_sources.shape[0]):
            freq_range, source_psd = scipy.signal.welch(centered_sources[idx, :], fs=self.sampling_frequency,
                                                        window="hamming", nperseg=segment_length,
                                                        nfft=nfft_length)

            idx = [x for x, _ in enumerate(freq_range) if x > self.frequency_emg][0]
            eeg_band_power = numpy.mean(source_psd[:idx])
            emg_band_power = numpy.mean(source_psd[idx:])
            ratio = eeg_band_power / emg_band_power
            self.psd_ratio.append(ratio)

        self.psd_ratio = numpy.array(self.psd_ratio)

    def remove_low_ratio_sources(self, sources: numpy.array) -> numpy.array:
        """
        Remove EMG components according to their PSD ratio
        :param sources: matrix with the eeg sources [sources x samples]
        """
        sorted_indices = numpy.argsort(self.psd_ratio)
        self.psd_ratio = self.psd_ratio[sorted_indices]

        limit_index = 0
        for i in self.psd_ratio:
            if i > self.ratio:
                break

        if limit_index < self.sources_tol[0]:
            limit_index = self.sources_tol[0]
        elif limit_index > self.sources_tol[1]:
            limit_index = self.sources_tol[1]

        logging.info(f"Number of sources to be removed = {limit_index}")
        for idx in sorted_indices[:limit_index]:
            sources[idx, :] = 0
        return sources

    def apply_by_segments(self, eeg_array: numpy.array) -> Tuple[numpy.array, numpy.array]:
        """
        It removes the EMG sources by splitting the eeg recording
        into smaller windows. Each window is processed independently.
        :param eeg_array: [channels x samples]
        """
        segments = []
        segment_lenght = self.segment_lenght * self.sampling_frequency
        next_idx = segment_lenght

        for previous_idx in range(0, eeg_array.shape[1], segment_lenght):
            segment = eeg_array[:, previous_idx: next_idx]
            segments.append(segment)
            next_idx += segment_lenght

        clean_eegs = []
        aggregated_sources = []
        for segment in segments:
            sources = self.fit_canonical_correlation(segment)
            self.fit_psd_ratio(sources)
            clean_sources = self.remove_low_ratio_sources(sources)
            clean_eeg = self.ccanalysis.inverse_transform(clean_sources)
            clean_eegs.append(clean_eeg)
            aggregated_sources.append(sources)

        return numpy.concat(aggregated_sources, axis=1), numpy.concat(clean_eegs, axis=1)
