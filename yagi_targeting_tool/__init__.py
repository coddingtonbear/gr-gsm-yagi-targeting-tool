from __future__ import print_function

import argparse
import csv
import datetime
import logging
import os
import sys
import tempfile

import click
import coloredlogs
import grgsm
import numpy

from .grgsm_scanner import wideband_scanner


logger = logging.getLogger(__name__)


PROVIDER_TABLE = {}


def sample_rate_type(raw_value):
    value = float(raw_value)
    if (value / 0.2e6) % 2 != 0:
        raise argparse.ArgumentTypeError(
            'Sample rate must be an even multiple of 0.2e6'
        )

    return value


def get_eta_string(started, current, total):
    if current == 0:
        # Assume 10-seconds per scan; just to have some kind of estimate
        estimated_remaining_time = (
            datetime.timedelta(seconds=10) * total
        )
    else:
        estimated_remaining_time = (
            (datetime.datetime.now() - started) / current * (total - current)
        )
    return '%sm %ss; %s' % (
        int(estimated_remaining_time.total_seconds()) // 60,
        int(estimated_remaining_time.total_seconds()) % 60,
        (datetime.datetime.now() + estimated_remaining_time).strftime(
            '%H:%M:%S'
        )
    )


def get_provider_table():
    global PROVIDER_TABLE

    if not PROVIDER_TABLE:
        with open(
            os.path.join(
                os.path.dirname(__file__),
                'mcc-mnc-table.csv',
            ),
            'r'
        ) as inf:
            reader = csv.DictReader(inf)
            for line in reader:
                try:
                    mcc = int(line['MCC'])
                    mnc = int(line['MNC'])
                    network = line['Network']
                except ValueError:
                    continue

                if mcc not in PROVIDER_TABLE:
                    PROVIDER_TABLE[mcc] = {}
                PROVIDER_TABLE[mcc][mnc] = network

    return PROVIDER_TABLE


class capture_stdout(object):
    def __enter__(self):
        self._stdout = tempfile.mkstemp()[0]
        self._stderr = tempfile.mkstemp()[0]
        self._save = os.dup(1), os.dup(2)
        os.dup2(self._stdout, 1)
        os.dup2(self._stderr, 2)

        return self

    def __exit__(self, *args):
        os.dup2(self._save[0], 1)
        os.dup2(self._save[1], 2)

        stdout_f = os.fdopen(self._stdout, 'r')
        stdout_f.seek(0)
        self.stdout = stdout_f.read()
        stdout_f.close()

        stderr_f = os.fdopen(self._stderr, 'r')
        stderr_f.seek(0)
        self.stderr = stderr_f.read()
        stderr_f.close()


class CellInfo(object):
    def __init__(self, **kwargs):
        for keyword, value in kwargs.items():
            setattr(self, keyword, value)

        self.measurements = {}

    def add_measurement(self, degrees, power):
        if degrees not in self.measurements:
            self.measurements[degrees] = []

        self.measurements[degrees].append(power)

    @property
    def network(self):
        providers = get_provider_table()

        return providers.get(self.mcc, {}).get(self.mnc, 'Unknown Network')

    def get_measurements(self):
        average_power = {}

        for direction, measurements in self.measurements.items():
            average_power[direction] = (
                float(sum(measurements)) / len(measurements)
            )

        return average_power

    def get_average_power(self, total_directions=None):
        measurements = []

        for _, measurement in self.get_measurements().items():
            measurements.append(measurement)

        if total_directions is not None:
            add_zeros = total_directions - len(measurements)
            measurements += [0] * add_zeros

        return float(sum(measurements)) / len(measurements)

    def __str__(self):
        return (
            "%s (ARFCN: %s, Freq: %s, "
            "CID: %s, LAC: %s, MCC: %s, MNC: %s)" % (
                self.network,
                self.arfcn,
                self.frequency/1e6,
                self.cell_id,
                self.lac,
                self.mcc,
                self.mnc
            )
        )

    def __unicode__(self):
        return unicode(self.__str__())

    def __repr__(self):
        return '<CellInfo %s>' % self.__str__()


def scan(
    band,
    min_frequency=0,
    max_frequency=float('infinity'),
    ppm=0,
    gain=0,
    if_gain=0,
    bb_gain=0,
    sample_rate=2e6,
    directions=4,
    osmocom_args=''
):
    channels_num = int(sample_rate / 0.2e6)
    start = grgsm.arfcn.arfcn2downlink(
        grgsm.arfcn.get_first_arfcn(band) + int(channels_num / 2) - 1,
        band
    )
    end = grgsm.arfcn.arfcn2downlink(
        grgsm.arfcn.get_last_arfcn(band) - int((channels_num / 2) - 1),
        band
    )
    stop = end + 0.2e6

    found_cells = {}

    freq_offsets = numpy.fft.ifftshift(
        numpy.array(
            range(
                int(-numpy.floor(channels_num/2)),
                int(numpy.floor((channels_num+1)/2))
            )
        ) * 2e5
    )

    for direction in range(0, 360, 360 / directions):
        current = start

        yes = click.confirm(
            u'Please point your antenna toward %s\u00b0.' % direction,
            default=True
        )
        if not yes:
            continue

        all_frequencies = range(
            int(start),
            int(stop),
            int(channels_num * 0.2e6)
        )
        started = datetime.datetime.now()
        for f_idx, current in enumerate(all_frequencies):
            if current < min_frequency or current > max_frequency:
                logger.debug(
                    u'Skipping scan of %s (%s); %s frequency.',
                    current,
                    direction,
                    (
                        'below minimum'
                        if current < min_frequency
                        else 'above maximum'
                    )
                )
                continue

            logger.info(
                u'Scanning %s\u00b0 on %s Mhz (eta %s)...',
                direction,
                round(current / 1e6, 1),
                get_eta_string(started, f_idx, len(all_frequencies))
            )

            with capture_stdout() as captured:
                scanner = wideband_scanner(
                    rec_len=5,
                    sample_rate=sample_rate,
                    carrier_frequency=current,
                    gain=gain,
                    if_gain=if_gain,
                    bb_gain=bb_gain,
                    ppm=ppm,
                    offset=0,
                    args=osmocom_args,
                )
                scanner.start()
                scanner.wait()
                scanner.stop()

            for line in captured.stdout.split('\n'):
                if line.strip():
                    logger.debug('Captured stdout: %s', line)
            for line in captured.stderr.split('\n'):
                if line.strip():
                    logger.debug('Captured stderr: %s', line)

            detected_channels = scanner.system_info.get_chans()

            if detected_channels:
                channels = numpy.array(scanner.system_info.get_chans())
                found_frequencies = current + freq_offsets[(channels)]

                cell_ids = numpy.array(scanner.system_info.get_cell_id())
                lacs = numpy.array(scanner.system_info.get_lac())
                mccs = numpy.array(scanner.system_info.get_mcc())
                mncs = numpy.array(scanner.system_info.get_mnc())
                ccch_confs = numpy.array(scanner.system_info.get_ccch_conf())

                powers = numpy.array(scanner.system_info.get_pwrs())

                for idx, channel in enumerate(channels):
                    cell_arfcn_list = (
                        scanner.system_info.get_cell_arfcns(channel)
                    )
                    neighbour_list = (
                        scanner.system_info.get_neighbours(channel)
                    )

                    cell_id = cell_ids[idx]
                    if cell_id not in found_cells:
                        found_cells[cell_id] = CellInfo(
                            arfcn=grgsm.arfcn.downlink2arfcn(
                                found_frequencies[idx],
                                band
                            ),
                            frequency=found_frequencies[idx],
                            cell_id=cell_id,
                            lac=lacs[idx],
                            mcc=mccs[idx],
                            mnc=mncs[idx],
                            ccch_conf=ccch_confs[idx],
                            neighbors=neighbour_list,
                            cell_arfcns=cell_arfcn_list
                        )
                    cell = found_cells[cell_id]
                    cell.add_measurement(direction, powers[idx])
                    logger.info(cell)

            # Allow the block to be garbage collected so it'll free
            # the RTLSDR device
            scanner = None

    return found_cells


def cmdline(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-b',
        '--band',
        dest='band',
        default='P-GSM',
        help=(
            'Specify the GSM band to scan through for matching towers.\n'
            'Available bands are: %s' % (
                ", ".join(grgsm.arfcn.get_bands())
            )
        ),
        choices=grgsm.arfcn.get_bands(),
    )
    parser.add_argument(
        '-p',
        '--ppm',
        dest='ppm',
        type=int,
        default=0,
        help=(
            'Frequency correction in parts-per-million.'
        )
    )
    parser.add_argument(
        '-d',
        '--directions',
        dest='directions',
        type=int,
        default=4,
        help=(
            'Number of directions to sample from; using a setting of '
            'four (N, E, S, W) is recommended.'
        )
    )
    parser.add_argument(
        '--min-frequency',
        dest='min_frequency',
        type=float,
        default=0,
        help=(
            'Minimum frequency to include in scan within a given band.'
        )
    )
    parser.add_argument(
        '--max-frequency',
        dest='max_frequency',
        type=float,
        default=float('infinity'),
        help=(
            'Maximum frequency to include in scan within a given band.'
        )
    )
    parser.add_argument(
        '--log-level',
        dest='log_level',
        default='INFO',
    )
    parser.add_argument(
        '-g',
        '--gain',
        default=24,
        type=float,
    )
    parser.add_argument(
        '--if-gain',
        default=None,
        type=int,
    )
    parser.add_argument(
        '--bb-gain',
        default=None,
        type=int,
    )
    parser.add_argument(
        '-s',
        '--sample-rate',
        default=2e6,
        type=sample_rate_type,
        help='Sample rate; must be an even multiple of 0.2e6.',
    )
    args = parser.parse_args(args)

    logger.setLevel(logging.getLevelName(args.log_level))
    logger_stream = logging.StreamHandler()
    logger_stream.setFormatter(
        coloredlogs.ColoredFormatter(
            fmt='%(asctime)s %(name)s %(levelname)s %(message)s',
        )
    )
    logger.addHandler(logger_stream)

    found_cells = scan(
        band=args.band,
        min_frequency=args.min_frequency,
        max_frequency=args.max_frequency,
        directions=args.directions,
        ppm=args.ppm,
        sample_rate=args.sample_rate,
        gain=args.gain if args.gain else None,
        if_gain=args.if_gain,
        bb_gain=args.bb_gain,
    )

    networks = {}

    for cell in found_cells.values():
        if cell.network not in networks:
            networks[cell.network] = []

        networks[cell.network].append(cell)

    for network, cells in networks.items():
        for cell in sorted(
            cells,
            key=lambda x: x.get_average_power(args.directions),
            reverse=True,
        ):
            logger.info('%s', cell)
            for direction, power in sorted(
                cell.get_measurements().items(),
                key=lambda x: x[1],
                reverse=True
            ):
                logger.info(u'- %s\u00b0: %s', direction, power)


if __name__ == '__main__':
    cmdline()
