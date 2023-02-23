# This script loads epigenetic data into memory
# It does this for multiple ChIP-seq assays
# It then also loads genome-wide RNA-seq data binned at 25bp resolution
# It does so for multiple cell types
# It then creates a generator
# which contains a masking function for the epigenetic data
# in order to perform masked-language modeling or supervised RNA-seq prediction
import numpy as np
from os.path import isfile
from tensorflow.python.keras.utils.data_utils import Sequence
import sys
import os
from random import randrange
import pyranges as pr


ALL_CELL_TYPES = ["T" + "{0:0=2d}".format(i) for i in range(1, 14)]

CELL_TYPES = ["T" + "{0:0=2d}".format(i) for i in range(1, 14)]
ASSAY_TYPES = ["A" + "{0:0=2d}".format(i) for i in range(2, 8)]
ACTIVE_ASSAY_TYPES = ["A" + "{0:0=2d}".format(i) for i in range(2, 8)]

training_chroms = ["chr"+str(i) for i in range(1, 14, 1)]
validation_chroms = ["chr"+str(i) for i in range(14, 23, 2)]
testing_chroms = ["chr"+str(i) for i in range(15, 23, 2)]

# Add stochasticity to chromosome choices as per Yun's suggestion 22/6/22
training_chroms = np.random.choice(training_chroms, 10, replace=False)
validation_chroms = np.random.choice(validation_chroms, 3, replace=False)
testing_chroms = np.random.choice(testing_chroms, 4, replace=False)

# Remove CXCR4 and TGFBR1 chromosomes from training and add them to validation and testing
'''
if("chr2" in training_chroms):
    training_chroms.remove("chr2")
if("chr9" in training_chroms):
    training_chroms.remove("chr9")
testing_chroms = testing_chroms + ["chr2", "chr9"]
'''

# testing_chroms = training_chroms # TEMPORARY FOR UNDERSTANDING PERFORMANCE METRICS ON TRAINING DATA

inference_chroms = ["chr"+str(i) for i in [2, 9]]

DEBUG = False
PRINT_FEATURES = False

EPS = 0.000001

MASK_VALUE = -10

RESOLUTION = 25

EDGE_CUSHION = 1000  # corresponds to 100Kb from the edge of chromosomes

if(RESOLUTION == 25):
    DATA_FOLDER = '../../Data/Transformed_25bp_Data'
    TRANSCRIPTOME_DATA_FOLDER = "../../Data/Gene_Expression_Data/" \
                                "25bp_genome_wide_TPM_npy"
else:
    print("RESOLUTION has to be 25bp!")
    sys.exit(-2)

TSS_DATA = "../../Data/Gene_Expression_Data/" \
           "T01.tsv.TPM.headered"

# We don't want to train in regions that are Blacklisted or have Gaps
# Blacklisted_Regions = pr.read_bed('../Data/hg38.Blacklisted.bed', as_df=False)
# Gap_Regions = pr.read_bed('../Data/hg38.Gaps.bed', as_df=False)

if(PRINT_FEATURES):
    f_output = open("../../Data/Training_Data.csv", 'w')


def check_region(chrom, start, end):
    start = int(start*1.0 * RESOLUTION)
    end = int(end*1.0 * RESOLUTION)

    if(len(Blacklisted_Regions[chrom, start:end]) +
       len(Gap_Regions[chrom, start:end]) > 0):
        return "bad"
    else:
        return "good"


def preprocess_epigenome(epigenome):

    return np.log1p(epigenome)


def create_masked(y, p):

    # dimensions are window_size x len(ASSAY_TYPES)
    # We mask out some positions in x and mask the opposite ones in y
    counter = 0
    x = np.copy(y)
    for i in range(x.shape[0]):
        if(np.random.uniform(low=0.0, high=1.0) < p):
            counter += 1
            x[i, :] = MASK_VALUE
        else:
            y[i, :] = MASK_VALUE

    if(DEBUG):
        # This can be used to debug how many entries are masked
        if(counter == 0):
            print("No entries have been masked", file=sys.stderr)
        elif(counter == x.shape[0]):
            print("All entries have been masked", file=sys.stderr)

    return x, y


class EpigenomeGenerator(Sequence):

    def __init__(self, window_size, batch_size,
                 shuffle=True, mode='', masking_probability=0.0, special_cell_type=None, chrom=None):

        self.window_size = window_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode
        self.masking_probability = masking_probability

        self.special_cell_type = special_cell_type

        self.epigenome = {}
        self.chrom_lens = {}
        self.transcriptome_pos = {}
        self.transcriptome_neg = {}

        if("training" in self.mode):
            self.chroms = training_chroms

        elif("validation" in self.mode):
            self.chroms = validation_chroms

        elif("testing" in self.mode):
            self.chroms = testing_chroms

        elif("inference" in self.mode):
            self.chroms = [chrom] # inference_chroms

            global ASSAY_TYPES
            global ACTIVE_ASSAY_TYPES
            ASSAY_TYPES = ["A" + "{0:0=2d}".format(i) for i in range(2, 8)] + ["A10"]
            ACTIVE_ASSAY_TYPES = ["A" + "{0:0=2d}".format(i) for i in range(2, 8)] + ["A10"]

        # Load TSS data
        self.TSS = []

        # Load TSS data
        if(not(isfile(TSS_DATA))):
            print(TSS_DATA, "doesn't exist!", file=sys.stderr)

        f_TSS = open(TSS_DATA, 'r')
        print("Loading TSS data", file=sys.stderr)

        line_number = 0
        for line in f_TSS:
            line_number += 1
            vec = line.rstrip("\n").split("\t")
            chrom = vec[0]
            TSS_strand = vec[3]
            transcript = vec[4]

            # Load TSS data only corresponding to the mode's chromosomes
            if(chrom not in self.chroms):
                continue

            for cell_type in CELL_TYPES:
        
                '''
                # This block of code is only for cross-cell_type generalization
                # where we would train on all but the special cell types
                ###############################################################        
                if(cell_type == self.special_cell_type):
                    if( ("training" in self.mode) or
                        ("validation" in self.mode) ):
                        continue
                ###############################################################        
                '''        

                if(TSS_strand == "+"):
                    self.TSS.append([chrom, vec[1], "+", cell_type, transcript])
                elif(TSS_strand == "-"):
                    self.TSS.append([chrom, vec[2], "-", cell_type, transcript])
                else:
                    if(line_number > 1):
                        print("TSS strand information in invalid",
                              file=sys.stderr)
                    else:
                        print("Parsing TSS file header",
                              file=sys.stderr)
                    continue

        for chrom in self.chroms:
            for cell_type in CELL_TYPES:

                epigenome = []
                for assay_type in ASSAY_TYPES:

                    print(cell_type, assay_type, chrom, file=sys.stderr)

                    f_name = cell_type+""+assay_type+"."+chrom+".npy"
                    f_name = DATA_FOLDER+"/"+f_name
                    if(isfile(f_name)):
                        print("Loading Epigenome data", f_name,
                              file=sys.stderr)

                        if(assay_type in ACTIVE_ASSAY_TYPES):
                            current_epigenome = np.load(f_name)
                            current_epigenome = preprocess_epigenome(
                                                current_epigenome)
                        else:
                            # Zero out NON-ACTIVE ASSAY_TYPES
                            current_epigenome = 0.0 * np.load(f_name)

                        epigenome.append(current_epigenome)
                    else:
                        print(assay_type, "missing in", cell_type, chrom)
                        sys.exit(-1)

                # Load transcriptome
                f_transcriptome_pos = cell_type + "_TPM." + chrom + ".+.npy"
                f_transcriptome_pos = (TRANSCRIPTOME_DATA_FOLDER + "/"
                                       + f_transcriptome_pos)

                f_transcriptome_neg = cell_type + "_TPM." + chrom + ".-.npy"
                f_transcriptome_neg = (TRANSCRIPTOME_DATA_FOLDER + "/"
                                       + f_transcriptome_neg)

                if(isfile(f_transcriptome_pos) and
                   isfile(f_transcriptome_neg)):
                    print("Loading Transcriptome data", f_transcriptome_pos,
                          f_transcriptome_neg, file=sys.stderr)
                    transcriptome_pos = np.load(f_transcriptome_pos)
                    transcriptome_neg = np.load(f_transcriptome_neg)
                else:
                    print("Transcriptome data missing", f_transcriptome_pos,
                          f_transcriptome_neg, file=sys.stderr)
                    transcriptome_pos = np.asarray([0])
                    transcriptome_neg = np.asarray([0])

                if(chrom not in self.epigenome):
                    self.epigenome[chrom] = {}
                    self.chrom_lens[chrom] = current_epigenome.shape[0]
                    self.transcriptome_pos[chrom] = {}
                    self.transcriptome_neg[chrom] = {}
                epigenome = np.vstack(epigenome)  # concatenate all assay types
                self.epigenome[chrom][cell_type] = epigenome
                self.transcriptome_pos[chrom][cell_type] = transcriptome_pos
                self.transcriptome_neg[chrom][cell_type] = transcriptome_neg

        # Now we need a way to randomly sample from the genome
        # For this we need chromosome lengths
        # We build a mapping from indexes to (chrom, position_in_chrom)
        self.chrom_list, self.tot_len_list = zip(*self.chrom_lens.items())
        self.tot_len_list = np.array(self.tot_len_list)
        self.tot_len_list = np.cumsum(self.tot_len_list)
        self.idxs = np.arange(self.tot_len_list[-1])

    # Apparently Keras doesn't call this at the end of every epoch!!!
    def on_epoch_end(self):

        if self.shuffle:
            np.random.shuffle(self.idxs)

    # So we have to call shuffle inside len which is called after every epoch!
    def __len__(self):

        if self.shuffle:
            np.random.shuffle(self.idxs)

        return self.tot_len_list[-1] // self.batch_size

    def idx_to_chrom_and_start(self, idx):

        chr_idx = np.where(self.tot_len_list > idx)[0][0]
        chrom = self.chrom_list[chr_idx]

        d = -1
        if(chr_idx == 0):
            d = idx
        else:
            d = idx - self.tot_len_list[chr_idx - 1]
        start = d

        return chrom, start

    def __getitem__(self, batch_number):

        X = np.zeros((self.batch_size, self.window_size, len(ASSAY_TYPES)))
        Y = np.zeros((self.batch_size, self.window_size, len(ASSAY_TYPES)))

        number_of_data_points = self.batch_size
        while(number_of_data_points > 0):
            random_idx = randrange(self.tot_len_list[-1])
            idx = self.idxs[random_idx]
            chrom, start = self.idx_to_chrom_and_start(idx)
            end = start + self.window_size

            if(DEBUG):
                print("Batch Number", batch_number, chrom, start, end,
                      file=sys.stderr)

            if((start < EDGE_CUSHION) or
               (end > self.chrom_lens[chrom] - EDGE_CUSHION)):
                # We are too close to the edges of the chromosome
                if(DEBUG):
                    print("We are too close to the edge!",
                          batch_number, idx, chrom, start,
                          end, self.chrom_lens[chrom],
                          file=sys.stderr)
                continue
            # TODO: This slows down training significantly
            # elif(check_region(chrom, start, end) == "bad"):
                # The training data point either lies in
                # a Blacklisted Region or a Gap Region in hg38
                # So we create the i'th data point to be a dummy with all 0s
                # Since X and Y are aleady 0s, we do nothing
                # if(DEBUG):
                #     print("Data point in Blacklisted or Gap region",
                #           batch_number, idx, chrom, start, d,
                #           end, self.chrom_lens[chrom],
                #           file=sys.stderr)
                # continue
            else:

                # Define generator for MLM
                if(("training" in self.mode) or ("validation" in self.mode)):
                    # Randomly sample a cell type
                    random_cell_type_index = randrange(len(CELL_TYPES))
                    if(DEBUG):
                        print("Sampled cell type", random_cell_type_index,
                              "for training", file=sys.stderr)
                else:
                    random_cell_type_index = 0  # Fix cell type for testing

                random_cell_type = CELL_TYPES[random_cell_type_index]

                # TODO: remove this transpose
                # TODO: add assert on size to make sure it's always consistent
                y = self.epigenome[chrom][random_cell_type][:, start:end]
                y = np.transpose(y)

                x_masked, y_masked = create_masked(y, self.masking_probability)
                # print(x_masked, y_masked)

                if(DEBUG):
                    if(x_masked.shape[0] != self.window_size):
                        print("Found the wrong shape!",
                              chrom, start, end, x_masked.shape,
                              y_masked.shape, file=sys.stderr)

                X[number_of_data_points-1] = x_masked
                Y[number_of_data_points-1] = y_masked

                number_of_data_points -= 1

        return X, Y


class TranscriptomeGenerator(EpigenomeGenerator):

    def __init__(self, window_size, batch_size,
                 shuffle=True, mode='', masking_probability=0.0,
                 cell_type_index=None, chrom=None):

        self.window_size = window_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode
        self.masking_probability = masking_probability

        self.special_cell_type = CELL_TYPES[cell_type_index]
        self.chrom = chrom

        self.log10p1_tpm_lower_bound = -1000000 
        self.log10p1_tpm_upper_bound = +1000000

        EpigenomeGenerator.__init__(self,
                                    self.window_size,
                                    self.batch_size,
                                    self.shuffle,
                                    self.mode,
                                    self.masking_probability,
                                    self.special_cell_type,
                                    self.chrom)
        
        self.idxs = np.arange(len(self.TSS))
    
    def __len__(self):

        if self.shuffle:
            np.random.shuffle(self.idxs)

        return len(self.idxs) // self.batch_size


    def __getitem__(self, batch_number):

        if(DEBUG):
            print("Batch number is", batch_number, 
                  "for mode:", self.mode, 
                  file=sys.stderr)

        X = np.zeros((self.batch_size, self.window_size, len(ASSAY_TYPES)))
        Y = np.zeros((self.batch_size, 1))
        metadata = []

        number_of_data_points = 0
        while(number_of_data_points < self.batch_size):

            if("genome_wide" in self.mode):
                genome_wide = True
            else:
                genome_wide = False

            # Based on Jeff and Kishore's suggestion
            # we want each batch to have atleast 20% TSS training
            random_number = randrange(self.batch_size)
            if((random_number > int(self.batch_size/5))
               and (genome_wide)
               and ('training' in self.mode)):

                random_idx = randrange(self.tot_len_list[-1])
                idx = self.idxs[random_idx]
                chrom, start = self.idx_to_chrom_and_start(idx)
                # Flip a coin to choose the strand if training genome wide
                random_toss = randrange(1)
                if(random_toss == 1):
                    strand = "+"
                else:
                    strand = "-"

            elif(("training" in self.mode) or
                 ("validation" in self.mode) or
                 ("testing" in self.mode)):

                # First obtain an index into a gene's TSS
                TSS_idx = self.idxs[batch_number * self.batch_size + 
                                   number_of_data_points]

                # What happens when we're out of TSSs?                
                if(TSS_idx >= len(self.TSS)):

                    print("TSS_idx shouldn't be larger than #genes",
                          batch_number, TSS_idx,
                          len(self.TSS), "genes", file=sys.stderr)
                    os._exit(1)
                
                chrom, start, strand, cell_type, transcript = self.TSS[TSS_idx]

                metadata.append([chrom, start, strand, cell_type, transcript])

                if(DEBUG):

                    print("batch_number =", batch_number,
                          "number_of_data_points =", number_of_data_points,
                          "TSS_idx =", TSS_idx,
                          "chrom =", chrom,
                          "start =", start,
                          "strand =", strand,
                          "cell type=", cell_type,
                          file=sys.stderr)

                start = int(int(start)/RESOLUTION)

            else:

                print("Mode is incorrect", file=sys.stderr)
                os._exit(-1)
                # random_idx = randrange(len(self.TSS))
                # chrom, start, strand = self.TSS[random_idx]
                # start = int(int(start)/RESOLUTION)
            
            # Skip chromosomes that don't correspond to this mode
            if("training" in self.mode):

                if(chrom not in training_chroms):

                    if(DEBUG):

                        print("Skipping chrom =", chrom,
                              "not in training chromosomes", file=sys.stderr)
                    continue

            elif("validation" in self.mode):

                if(chrom not in validation_chroms):

                    if(DEBUG):

                        print("Skipping chrom =", chrom,
                              "not in validation chromosomes", file=sys.stderr)
                    continue

            elif("testing" in self.mode):

                if(chrom not in testing_chroms):

                    if(DEBUG):

                        print("Skipping chrom =", chrom,
                              "not in testing chromosomes", file=sys.stderr)
                    continue

            end = start + self.window_size

            if(DEBUG):

                print("Batch Number", batch_number, chrom, start, end,
                      file=sys.stderr)

            if((start < EDGE_CUSHION) or
               (end > self.chrom_lens[chrom] - EDGE_CUSHION)):

                # We are too close to the edges of the chromosome
                if(DEBUG):

                    print("We are too close to the edge!",
                          batch_number, idx, chrom, start,
                          end, self.chrom_lens[chrom],
                          file=sys.stderr)
                continue
            # TODO: This slows down training significantly
            # elif(check_region(chrom, start, end) == "bad"):
                # The training data point either lies in
                # a Blacklisted Region or a Gap Region in hg38
                # So we create the i'th data point to be a dummy with all 0s
                # Since X and Y are aleady 0s, we do nothing
                # if(DEBUG):
                #     print("Data point in Blacklisted or Gap region",
                #           batch_number, idx, chrom, start, d,
                #           end, self.chrom_lens[chrom],
                #           file=sys.stderr)
                # continue
            else:

                if(DEBUG):
                    print(self.epigenome[chrom][cell_type].shape, start,
                          file=sys.stderr)

                x = (self.epigenome[chrom][cell_type]
                                   [:,
                                    start - (self.window_size // 2):
                                    start + (self.window_size // 2) + 1])
                x = np.transpose(x)

                if(True):

                    if(DEBUG):

                        print(self.transcriptome_pos[chrom][cell_type].shape,
                              start, file=sys.stderr)

                    if(strand == "+"):

                        y = (self.transcriptome_pos[chrom]
                                                   [cell_type]
                                                   [start])
                    else:

                        x = x[::-1, :]
                        y = (self.transcriptome_neg[chrom]
                                                   [cell_type]
                                                   [start])

                # TODO: 1 December 2021
                # We lose ~1k genes because TSSs lie within RESOLUTION
                # For those, print out exact TSS in addition to binned TSS
                # RESOLVED: 17 April 2022: This probably no longer occurs because
                # each TSS is associated with a transcript as well as location
                # So this is resolved now
                if(PRINT_FEATURES):

                    for assay_index in range(2, 8):
                        feature = x[:, assay_index-2]

                        if(feature.shape[0] != self.window_size):

                            print(cell_type, chrom, start,
                                  strand, assay_index, feature.shape, feature,
                                  file=sys.stderr)
                            sys.exit(-3)

                        output_string = np.array2string(feature,
                                                        separator=',')
                        output_string = output_string.lstrip('[').rstrip(']')
                        output_string = output_string.replace('\n', '')
                        output_string = output_string.replace(', ', ',')
                        output_string = output_string.replace(' ,', ',')
                        output_list = output_string.split(",")

                        if(len(output_list) != self.window_size):

                            print(output_list, output_string,
                                  file=sys.stderr)

                        print(output_string, y, cell_type, chrom, start,
                              strand, assay_index, sep=',', file=f_output)

                # Train on only a certain range of TPM 
                if( ("training" in self.mode) or ("validation" in self.mode) ):
                    if(not( (y >= self.log10p1_tpm_lower_bound) and (y < self.log10p1_tpm_upper_bound) )):
                        continue
        
                # For the purpose of CRISPRability, we zero out all other tracks
                # x[:, np.r_[0,1,3,4,5]] = 0    

                X[number_of_data_points, :, :] = x
                Y[number_of_data_points, :] = y

                number_of_data_points += 1

        if( ("training" in self.mode) or ("validation" in self.mode) ):
            return X, Y
        elif("testing" in self.mode):
            return X, Y, metadata
        else:
            return X, Y


class TranscriptomePredictor(EpigenomeGenerator):

    def __init__(self, window_size, batch_size,
                 shuffle=False, mode='inference', masking_probability=0.,
                 chrom="chr1", start=1, strand="+",
                 cell_type_index=-1):

        self.window_size = window_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode
        self.masking_probability = masking_probability
        self.chrom = chrom
        self.start = start // RESOLUTION
        self.strand = strand
        self.special_cell_type = CELL_TYPES[cell_type_index]

        EpigenomeGenerator.__init__(self, self.window_size, self.batch_size,
                                    self.shuffle, self.mode,
                                    self.masking_probability, self.special_cell_type, self.chrom)

    def __getitem__(self, batch_number):

        X = np.zeros((self.batch_size, self.window_size, len(ASSAY_TYPES)))
        Y = np.zeros((self.batch_size, 1))

        for i in range(self.start, self.start+1):
            if(DEBUG):
                print("Batch Number", batch_number, i, self.chrom, self.start,
                      self.strand, file=sys.stderr)

            x = (self.epigenome[self.chrom]
                               [self.special_cell_type]
                               [:,
                                i - (self.window_size // 2):
                                i + (self.window_size // 2) + 1])
            x = np.transpose(x)

            if(self.strand == "+"):
                y = (self.transcriptome_pos[self.chrom]
                                           [self.special_cell_type]
                                           [i])
            else:
                x = x[::-1, :]
                y = (self.transcriptome_neg[self.chrom]
                                           [self.special_cell_type]
                                           [i])

            X[i-self.start, :, :] = x
            Y[i-self.start, :] = y

        return X, Y
