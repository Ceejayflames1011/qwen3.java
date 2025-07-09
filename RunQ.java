// Inference for Qwen-3 Transformer model in pure Java, int8 quantized forward pass.
// Converted from the original C code in runq.c by A. Karpathy.

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.*;
import java.util.stream.IntStream;

public class RunQ {

    // ----------------------------------------------------------------------------
    // Transformer model data structures

    static class Config {
        int magic_number;
        int version;
        int dim;
        int hidden_dim;
        int n_layers;
        int n_heads;
        int n_kv_heads;
        int vocab_size;
        int seq_len;
        int head_dim;
        boolean shared_classifier;
        int group_size;

        Config(ByteBuffer buffer) {
            this.magic_number = buffer.getInt();
            this.version = buffer.getInt();
            this.dim = buffer.getInt();
            this.hidden_dim = buffer.getInt();
            this.n_layers = buffer.getInt();
            this.n_heads = buffer.getInt();
            this.n_kv_heads = buffer.getInt();
            this.vocab_size = buffer.getInt();
            this.seq_len = buffer.getInt();
            this.head_dim = buffer.getInt();
            this.shared_classifier = buffer.getInt() == 1;
            this.group_size = buffer.getInt();
            
            // Consume remaining header bytes
            for(int i = 0; i < (256 - 12 * 4) / 4; i++) {
                buffer.getInt();
            }
        }
    }

    static class QuantizedTensor {
        byte[] q; // quantized values
        float[] s; // scaling factors

        QuantizedTensor(int size, int group_size) {
            this.q = new byte[size];
            this.s = new float[size / group_size];
        }
    }

    static class TransformerWeights {
        // token embedding table
        QuantizedTensor q_tokens;
        float[] token_embedding_table;
        // weights for rmsnorms
        float[] rms_att_weight;
        float[] rms_ffn_weight;
        // weights for matmuls
        QuantizedTensor[] wq;
        QuantizedTensor[] wk;
        QuantizedTensor[] wv;
        QuantizedTensor[] wo;
        // QK-RMSNorm for Qwen3
        float[] q_ln_weights;
        float[] k_ln_weights;
        // weights for ffn
        QuantizedTensor[] w1;
        QuantizedTensor[] w2;
        QuantizedTensor[] w3;
        // final rmsnorm
        float[] rms_final_weight;
        // (optional) classifier weights
        QuantizedTensor wcls;

        TransformerWeights(Config p, ByteBuffer buffer) {
            int GS = p.group_size;
            
            rms_att_weight = readFloats(buffer, p.n_layers * p.dim);
            rms_ffn_weight = readFloats(buffer, p.n_layers * p.dim);
            rms_final_weight = readFloats(buffer, p.dim);
            q_ln_weights = readFloats(buffer, p.n_layers * p.head_dim);
            k_ln_weights = readFloats(buffer, p.n_layers * p.head_dim);

            q_tokens = init_quantized_tensors(buffer, 1, p.vocab_size * p.dim, GS)[0];
            token_embedding_table = new float[p.vocab_size * p.dim];
            dequantize(q_tokens, token_embedding_table, p.vocab_size * p.dim, GS);
            
            int all_heads_dim = p.n_heads * p.head_dim;
            int kv_dim = p.n_kv_heads * p.head_dim;

            wq = init_quantized_tensors(buffer, p.n_layers, p.dim * all_heads_dim, GS);
            wk = init_quantized_tensors(buffer, p.n_layers, p.dim * kv_dim, GS);
            wv = init_quantized_tensors(buffer, p.n_layers, p.dim * kv_dim, GS);
            wo = init_quantized_tensors(buffer, p.n_layers, all_heads_dim * p.dim, GS);

            w1 = init_quantized_tensors(buffer, p.n_layers, p.dim * p.hidden_dim, GS);
            w2 = init_quantized_tensors(buffer, p.n_layers, p.hidden_dim * p.dim, GS);
            w3 = init_quantized_tensors(buffer, p.n_layers, p.dim * p.hidden_dim, GS);

            wcls = p.shared_classifier ? q_tokens : init_quantized_tensors(buffer, 1, p.dim * p.vocab_size, GS)[0];
        }

        private static float[] readFloats(ByteBuffer buffer, int size) {
            float[] arr = new float[size];
            buffer.asFloatBuffer().get(arr);
            buffer.position(buffer.position() + size * Float.BYTES);
            return arr;
        }

        private static QuantizedTensor[] init_quantized_tensors(ByteBuffer buffer, int n, int size_each, int GS) {
            QuantizedTensor[] res = new QuantizedTensor[n];
            for (int i = 0; i < n; i++) {
                res[i] = new QuantizedTensor(size_each, GS);
                buffer.get(res[i].q);
                buffer.asFloatBuffer().get(res[i].s);
                buffer.position(buffer.position() + res[i].s.length * Float.BYTES);
            }
            return res;
        }
    }

    static class RunState {
        float[] x;
        float[] xb;
        float[] xb2;
        float[] hb;
        float[] hb2;
        QuantizedTensor xq;
        QuantizedTensor hq;
        float[] q;
        float[] k; // A view into key_cache
        float[] v; // A view into value_cache
        float[] att;
        float[] logits;
        float[] key_cache;
        float[] value_cache;

        RunState(Config p) {
            int GS = p.group_size;
            int all_heads_dim = p.n_heads * p.head_dim;
            int kv_dim = p.n_kv_heads * p.head_dim;

            x = new float[p.dim];
            xb = new float[all_heads_dim];
            xb2 = new float[p.dim];
            hb = new float[p.hidden_dim];
            hb2 = new float[p.hidden_dim];
            xq = new QuantizedTensor(all_heads_dim, GS);
            hq = new QuantizedTensor(p.hidden_dim, GS);
            q = new float[all_heads_dim];
            att = new float[p.n_heads * p.seq_len];
            logits = new float[p.vocab_size];
            key_cache = new float[p.n_layers * p.seq_len * kv_dim];
            value_cache = new float[p.n_layers * p.seq_len * kv_dim];
        }
    }

    static class Transformer {
        Config config;
        TransformerWeights weights;
        RunState state;
        ByteBuffer data;
        long fileSize;

        Transformer(String checkpointPath, int ctx_length) throws IOException {
            try (FileChannel fc = FileChannel.open(Paths.get(checkpointPath), StandardOpenOption.READ)) {
                fileSize = fc.size();
                data = fc.map(FileChannel.MapMode.READ_ONLY, 0, fileSize);
                data.order(ByteOrder.LITTLE_ENDIAN);

                this.config = new Config(data);
                if (this.config.magic_number != 0x616a6331) {
                     throw new IOException("File " + checkpointPath + " is not a qwen3.c checkpoint");
                }
                if (this.config.version != 1) {
                     throw new IOException("Checkpoint " + checkpointPath + " is version " + this.config.version + ", need version 1");
                }
                if (ctx_length > 0 && ctx_length <= this.config.seq_len) {
                    this.config.seq_len = ctx_length;
                }

                this.weights = new TransformerWeights(this.config, data);
                this.state = new RunState(this.config);
            }
        }
    }

    // ----------------------------------------------------------------------------
    // Quantization functions

    static void dequantize(QuantizedTensor qx, float[] x, int n, int GS) {
        for (int i = 0; i < n; i++) {
            x[i] = qx.q[i] * qx.s[i / GS];
        }
    }

    static void quantize(QuantizedTensor qx, float[] x, int n, int GS) {
        int num_groups = n / GS;
        float Q_MAX = 127.0f;

        for (int group = 0; group < num_groups; group++) {
            float wmax = 0.0f;
            int base = group * GS;
            for (int i = 0; i < GS; i++) {
                float val = Math.abs(x[base + i]);
                if (val > wmax) {
                    wmax = val;
                }
            }

            float scale = wmax / Q_MAX;
            qx.s[group] = scale;

            for (int i = 0; i < GS; i++) {
                float quant_value = x[base + i] / scale;
                qx.q[base + i] = (byte) Math.round(quant_value);
            }
        }
    }

    // ----------------------------------------------------------------------------
    // Neural net blocks

    static void rmsnorm(float[] o, int o_offset, float[] x, int x_offset, float[] weight, int weight_offset, int size) {
        float ss = 0.0f;
        for (int j = 0; j < size; j++) {
            ss += x[x_offset + j] * x[x_offset + j];
        }
        ss /= size;
        ss += 1e-6f;
        ss = 1.0f / (float) Math.sqrt(ss);

        for (int j = 0; j < size; j++) {
            o[o_offset + j] = weight[weight_offset + j] * (ss * x[x_offset + j]);
        }
    }

    static void softmax(float[] x, int x_offset, int size) {
        if (size == 1) {
            x[x_offset] = 1.0f;
            return;
        }
        float max_val = x[x_offset];
        for (int i = 1; i < size; i++) {
            if (x[x_offset + i] > max_val) {
                max_val = x[x_offset + i];
            }
        }
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            x[x_offset + i] = (float) Math.exp(x[x_offset + i] - max_val);
            sum += x[x_offset + i];
        }
        for (int i = 0; i < size; i++) {
            x[x_offset + i] /= sum;
        }
    }

    static void matmul(float[] xout, QuantizedTensor x, QuantizedTensor w, int n, int d) {
        int GS = x.s.length > 0 ? x.q.length / x.s.length : 0;

        IntStream.range(0, d).parallel().forEach(i -> {
            float val = 0.0f;
            int in = i * n;
            for (int j = 0; j <= n - GS; j += GS) {
                int ival = 0;
                for (int k = 0; k < GS; k++) {
                    ival += x.q[j + k] * w.q[in + j + k];
                }
                val += ((float) ival) * w.s[(in + j) / GS] * x.s[j / GS];
            }
            xout[i] = val;
        });
    }

    static float[] forward(Transformer transformer, int token, int pos) {
        Config p = transformer.config;
        TransformerWeights w = transformer.weights;
        RunState s = transformer.state;
        float[] x = s.x;
        int dim = p.dim;
        int kv_dim = p.n_kv_heads * p.head_dim;
        int kv_mul = p.n_heads / p.n_kv_heads;
        int hidden_dim = p.hidden_dim;
        int all_heads_dim = p.n_heads * p.head_dim;
        int GS = p.group_size;

        System.arraycopy(w.token_embedding_table, token * dim, x, 0, dim);

        for (int l = 0; l < p.n_layers; l++) {
            long loff = (long) l * p.seq_len * kv_dim;
            
            s.k = s.key_cache; // Using s.k and s.v as pointers into the cache requires offsets
            int k_offset = (int) (loff + (long)pos * kv_dim);
            s.v = s.value_cache;
            int v_offset = (int) (loff + (long)pos * kv_dim);

            rmsnorm(s.xb, 0, x, 0, w.rms_att_weight, l * dim, dim);

            quantize(s.xq, s.xb, dim, GS);
            matmul(s.q, s.xq, w.wq[l], dim, all_heads_dim);
            
            // For k and v, matmul result goes directly into the cache via offsets
            // To do this without creating a temporary buffer, we'd need a matmul version that supports output offsets.
            // For simplicity, let's use a temporary buffer and then copy.
            float[] temp_k = new float[kv_dim];
            float[] temp_v = new float[kv_dim];
            matmul(temp_k, s.xq, w.wk[l], dim, kv_dim);
            matmul(temp_v, s.xq, w.wv[l], dim, kv_dim);
            
            for (int h = 0; h < p.n_heads; h++) {
                int q_h_offset = h * p.head_dim;
                rmsnorm(s.q, q_h_offset, s.q, q_h_offset, w.q_ln_weights, l * p.head_dim, p.head_dim);
                for (int j = 0; j < p.head_dim / 2; j++) {
                    float freq = (float) Math.pow(1e6, -(float) j / (p.head_dim / 2));
                    float cos_freq = (float) Math.cos(pos * freq);
                    float sin_freq = (float) Math.sin(pos * freq);
                    float q_real = s.q[q_h_offset + j];
                    float q_imag = s.q[q_h_offset + j + p.head_dim / 2];
                    s.q[q_h_offset + j] = q_real * cos_freq - q_imag * sin_freq;
                    s.q[q_h_offset + j + p.head_dim / 2] = q_real * sin_freq + q_imag * cos_freq;
                }
            }
            
            for (int h = 0; h < p.n_kv_heads; h++) {
                int k_h_offset = h * p.head_dim;
                rmsnorm(temp_k, k_h_offset, temp_k, k_h_offset, w.k_ln_weights, l * p.head_dim, p.head_dim);
                for (int j = 0; j < p.head_dim / 2; j++) {
                    float freq = (float) Math.pow(1e6, -(float) j / (p.head_dim / 2));
                    float cos_freq = (float) Math.cos(pos * freq);
                    float sin_freq = (float) Math.sin(pos * freq);
                    float k_real = temp_k[k_h_offset + j];
                    float k_imag = temp_k[k_h_offset + j + p.head_dim / 2];
                    temp_k[k_h_offset + j] = k_real * cos_freq - k_imag * sin_freq;
                    temp_k[k_h_offset + j + p.head_dim / 2] = k_real * sin_freq + k_imag * cos_freq;
                }
            }
            
            System.arraycopy(temp_k, 0, s.key_cache, k_offset, kv_dim);
            System.arraycopy(temp_v, 0, s.value_cache, v_offset, kv_dim);
            

            IntStream.range(0, p.n_heads).parallel().forEach(h -> {
                int q_offset = h * p.head_dim;
                int att_offset = h * p.seq_len;
                for (int t = 0; t <= pos; t++) {
                    int key_cache_offset = (int) (loff + (long)t * kv_dim + (h / kv_mul) * p.head_dim);
                    float score = 0.0f;
                    for (int i = 0; i < p.head_dim; i++) {
                        score += s.q[q_offset + i] * s.key_cache[key_cache_offset + i];
                    }
                    s.att[att_offset + t] = score / (float) Math.sqrt(p.head_dim);
                }

                softmax(s.att, att_offset, pos + 1);

                int xb_offset = h * p.head_dim;
                Arrays.fill(s.xb, xb_offset, xb_offset + p.head_dim, 0.0f);
                for (int t = 0; t <= pos; t++) {
                    int value_cache_offset = (int) (loff + (long)t * kv_dim + (h / kv_mul) * p.head_dim);
                    float a = s.att[att_offset + t];
                    for (int i = 0; i < p.head_dim; i++) {
                        s.xb[xb_offset + i] += a * s.value_cache[value_cache_offset + i];
                    }
                }
            });

            quantize(s.xq, s.xb, all_heads_dim, GS);
            matmul(s.xb2, s.xq, w.wo[l], all_heads_dim, dim);

            for (int i = 0; i < dim; i++) {
                x[i] += s.xb2[i];
            }

            rmsnorm(s.xb, 0, x, 0, w.rms_ffn_weight, l * dim, dim);

            quantize(s.xq, s.xb, dim, GS);
            matmul(s.hb, s.xq, w.w1[l], dim, hidden_dim);
            matmul(s.hb2, s.xq, w.w3[l], dim, hidden_dim);

            for (int i = 0; i < hidden_dim; i++) {
                float val = s.hb[i];
                val *= (1.0f / (1.0f + (float) Math.exp(-val)));
                val *= s.hb2[i];
                s.hb[i] = val;
            }

            quantize(s.hq, s.hb, hidden_dim, GS);
            matmul(s.xb, s.hq, w.w2[l], hidden_dim, dim);

            for (int i = 0; i < dim; i++) {
                x[i] += s.xb[i];
            }
        }

        rmsnorm(x, 0, x, 0, w.rms_final_weight, 0, dim);
        quantize(s.xq, x, dim, GS);
        matmul(s.logits, s.xq, w.wcls, dim, p.vocab_size);
        return s.logits;
    }

    // ----------------------------------------------------------------------------
    // BPE Tokenizer

    static class Tokenizer {
        String[] vocab;
        float[] merge_scores;
        int vocab_size;
        int max_token_length;
        int bos_token_id;
        int eos_token_id;
        String prompt_template;
        String system_prompt_template;
        Map<String, Integer> vocab_map;

        Tokenizer(String checkpoint_path, int vocab_size, boolean enable_thinking) throws IOException {
            this.vocab_size = vocab_size;
            this.vocab = new String[vocab_size];
            this.merge_scores = new float[vocab_size];
            this.vocab_map = new HashMap<>(vocab_size);

            String tokenizer_path = checkpoint_path + ".tokenizer";
            try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(tokenizer_path)))) {
                this.max_token_length = Integer.reverseBytes(dis.readInt());
                this.bos_token_id = Integer.reverseBytes(dis.readInt());
                this.eos_token_id = Integer.reverseBytes(dis.readInt());

                for (int i = 0; i < vocab_size; i++) {
                    try {
                        this.merge_scores[i] = Float.intBitsToFloat(Integer.reverseBytes(dis.readInt()));
                        int len = Integer.reverseBytes(dis.readInt());
                        byte[] bytes = new byte[len];
                        dis.readFully(bytes);
                        this.vocab[i] = new String(bytes);
                    } catch (EOFException e) {
                        this.vocab[i] = "";
                    }
                    if (this.vocab[i] != null && !this.vocab[i].isEmpty()) {
                        this.vocab_map.put(this.vocab[i], i);
                    }
                }
            }
            
            this.prompt_template = loadPromptTemplate(checkpoint_path, false, enable_thinking);
            this.system_prompt_template = loadPromptTemplate(checkpoint_path, true, enable_thinking);
        }
        
        private String loadPromptTemplate(String checkpointPath, boolean withSystem, boolean enableThinking) throws IOException {
             String suffix = "";
             if (withSystem) {
                 suffix = enableThinking ? ".template.with-system-and-thinking" : ".template.with-system";
             } else {
                 suffix = enableThinking ? ".template.with-thinking" : ".template";
             }
             File file = new File(checkpointPath + suffix);
             if (!file.exists()) {
                 throw new IOException("Could not load prompt template: " + file.getPath());
             }
             byte[] buffer = new byte[1024];
             try(InputStream is = new FileInputStream(file)) {
                 is.read(buffer);
             }
             // Find null terminator
             int len = 0;
             while(len < buffer.length && buffer[len] != 0) {
                 len++;
             }
             return new String(buffer, 0, len);
        }

        String decode(int token) {
            return vocab[token];
        }

        List<Integer> encode(String text) {
            List<Integer> tokens = new ArrayList<>();
            for (int i = 0; i < text.length(); ) {
                char c = text.charAt(i);
                int id;
                int end_of_token_pos = -1;
                
                if (c == '<') {
                    int endPos = text.indexOf('>', i);
                    if (endPos != -1) {
                         String special_token = text.substring(i, endPos + 1);
                         Integer special_id = vocab_map.get(special_token);
                         if (special_id != null) {
                             tokens.add(special_id);
                             i += special_token.length();
                             continue;
                         }
                    }
                }

                String s = String.valueOf(c);
                Integer found_id = vocab_map.get(s);
                if (found_id != null) {
                    tokens.add(found_id);
                } else {
                    System.err.println("Warning: unknown character in input: " + s);
                }
                i++;
            }

            while (true) {
                float best_score = -1e10f;
                int best_id = -1;
                int best_idx = -1;

                for (int i = 0; i < tokens.size() - 1; i++) {
                    String merged = vocab[tokens.get(i)] + vocab[tokens.get(i + 1)];
                    Integer id = vocab_map.get(merged);
                    if (id != null && merge_scores[id] > best_score) {
                        best_score = merge_scores[id];
                        best_id = id;
                        best_idx = i;
                    }
                }

                if (best_idx == -1) break;

                tokens.set(best_idx, best_id);
                tokens.remove(best_idx + 1);
            }
            return tokens;
        }
    }

    // ----------------------------------------------------------------------------
    // Sampler

    static class ProbIndex implements Comparable<ProbIndex> {
        float prob;
        int index;

        ProbIndex(float prob, int index) {
            this.prob = prob;
            this.index = index;
        }

        @Override
        public int compareTo(ProbIndex o) {
            return Float.compare(o.prob, this.prob);
        }
    }

    static class Sampler {
        int vocab_size;
        ProbIndex[] probindex;
        float temperature;
        float topp;
        Random rng;

        Sampler(int vocab_size, float temperature, float topp, long rng_seed) {
            this.vocab_size = vocab_size;
            this.temperature = temperature;
            this.topp = topp;
            this.rng = new Random(rng_seed);
            this.probindex = new ProbIndex[vocab_size];
        }

        int sample(float[] logits) {
            if (temperature == 0.0f) {
                return sample_argmax(logits);
            } else {
                for (int q = 0; q < vocab_size; q++) {
                    logits[q] /= temperature;
                }
                softmax(logits, 0, vocab_size);
                float coin = rng.nextFloat();
                if (topp <= 0 || topp >= 1) {
                    return sample_mult(logits, coin);
                } else {
                    return sample_topp(logits, topp, coin);
                }
            }
        }
        
        int sample_argmax(float[] probabilities) {
            int max_i = 0;
            float max_p = probabilities[0];
            for (int i = 1; i < vocab_size; i++) {
                if (probabilities[i] > max_p) {
                    max_i = i;
                    max_p = probabilities[i];
                }
            }
            return max_i;
        }

        int sample_mult(float[] probabilities, float coin) {
            float cdf = 0.0f;
            for (int i = 0; i < vocab_size; i++) {
                cdf += probabilities[i];
                if (coin < cdf) {
                    return i;
                }
            }
            return vocab_size - 1;
        }

        int sample_topp(float[] probabilities, float topp, float coin) {
            int n0 = 0;
            float cutoff = (1.0f - topp) / (vocab_size - 1);
            for (int i = 0; i < vocab_size; i++) {
                if (probabilities[i] >= cutoff) {
                    probindex[n0] = new ProbIndex(probabilities[i], i);
                    n0++;
                }
            }
            Arrays.sort(probindex, 0, n0);

            float cumulative_prob = 0.0f;
            int last_idx = n0 - 1;
            for (int i = 0; i < n0; i++) {
                cumulative_prob += probindex[i].prob;
                if (cumulative_prob > topp) {
                    last_idx = i;
                    break;
                }
            }

            float r = coin * cumulative_prob;
            float cdf = 0.0f;
            for (int i = 0; i <= last_idx; i++) {
                cdf += probindex[i].prob;
                if (r < cdf) {
                    return probindex[i].index;
                }
            }
            return probindex[last_idx].index;
        }
    }
    
    // ----------------------------------------------------------------------------
    // Generation loop

    public static void generate(Transformer transformer, Tokenizer tokenizer, Sampler sampler, String prompt) {
        if (prompt == null) prompt = "";

        List<Integer> prompt_tokens = tokenizer.encode(prompt);
        if (prompt_tokens.isEmpty()) {
            System.err.println("Please provide a prompt using -i <string> on the command line.");
            System.exit(1);
        }

        int token = prompt_tokens.get(0);
        int pos = 0;

        while (pos < transformer.config.seq_len) {
            float[] logits = forward(transformer, token, pos);

            int next_token;
            if (pos < prompt_tokens.size() - 1) {
                next_token = prompt_tokens.get(pos + 1);
            } else {
                next_token = sampler.sample(logits);
            }
            pos++;
            
            System.out.print(tokenizer.decode(token));
            System.out.flush();

            if (pos >= prompt_tokens.size() && (next_token == tokenizer.bos_token_id || next_token == tokenizer.eos_token_id)) {
                break;
            }
            token = next_token;
        }
        System.out.println();
    }
    
    // ----------------------------------------------------------------------------
    // Chat loop
    
    public static void chat(Transformer transformer, Tokenizer tokenizer, Sampler sampler, String cli_user_prompt, String system_prompt) {
        Scanner scanner = new Scanner(System.in);
        int pos = 0;
        boolean user_turn = true;
        List<Integer> prompt_tokens = null;
        int user_idx = 0;
        int next_token = -1;

        while (true) {
             if (pos >= transformer.config.seq_len) {
                System.out.println("\n(context window full, clearing)");
                pos = 0;
                user_turn = true;
            }
            
            if (user_turn) {
                String user_prompt;
                if (cli_user_prompt != null && pos == 0) {
                     user_prompt = cli_user_prompt;
                } else {
                     System.out.print("\n> ");
                     user_prompt = scanner.nextLine();
                     if (user_prompt == null || user_prompt.isEmpty()) {
                         break;
                     }
                }
                
                String rendered_prompt;
                if(pos == 0 && system_prompt != null && !system_prompt.isEmpty()) {
                    rendered_prompt = String.format(tokenizer.system_prompt_template, system_prompt, user_prompt);
                } else {
                    rendered_prompt = String.format(tokenizer.prompt_template, user_prompt);
                }
                
                prompt_tokens = tokenizer.encode(rendered_prompt);
                user_idx = 0;
                user_turn = false;
            }
            
            int current_token;
            if(user_idx < prompt_tokens.size()) {
                current_token = prompt_tokens.get(user_idx++);
            } else {
                current_token = next_token;
            }
            
            float[] logits = forward(transformer, current_token, pos);
            next_token = sampler.sample(logits);
            pos++;
            
            if (user_idx >= prompt_tokens.size()) {
                if (current_token == tokenizer.bos_token_id || current_token == tokenizer.eos_token_id) {
                    System.out.println();
                    user_turn = true;
                } else if (next_token != tokenizer.bos_token_id && next_token != tokenizer.eos_token_id) {
                    System.out.print(tokenizer.decode(next_token));
                    System.out.flush();
                }
            }
        }
        scanner.close();
    }
    
    // ----------------------------------------------------------------------------
    // CLI

    public static void error_usage() {
        System.err.println("Usage:   java RunQ <checkpoint> [options]");
        System.err.println("Example: java RunQ Qwen3-4B.bin -r 1");
        System.err.println("Options:");
        System.err.println("  -t <float>  temperature in [0,inf], default 1.0");
        System.err.println("  -p <float>  p value in top-p (nucleus) sampling in [0,1], default 0.9");
        System.err.println("  -s <int>    random seed, default time(NULL)");
        System.err.println("  -c <int>    context window size, 0 (default) = max_seq_len");
        System.err.println("  -m <string> mode: generate|chat, default: chat");
        System.err.println("  -i <string> input prompt");
        System.err.println("  -y <string> system prompt in chat mode, default is none");
        System.err.println("  -r <int>    reasoning mode, 0 (default) = no thinking, 1 = thinking");
        System.exit(1);
    }
    
    public static void main(String[] args) {
        String checkpoint_path = null;
        float temperature = 1.0f;
        float topp = 0.9f;
        String prompt = null;
        long rng_seed = 0;
        String mode = "chat";
        String system_prompt = null;
        boolean enable_thinking = false;
        int ctx_length = 0;

        if (args.length < 1) {
            error_usage();
        }
        checkpoint_path = args[0];

        for (int i = 1; i < args.length; i += 2) {
            if (i + 1 >= args.length) { error_usage(); }
            String flag = args[i];
            String value = args[i+1];
            switch (flag) {
                case "-t": temperature = Float.parseFloat(value); break;
                case "-p": topp = Float.parseFloat(value); break;
                case "-s": rng_seed = Long.parseLong(value); break;
                case "-c": ctx_length = Integer.parseInt(value); break;
                case "-i": prompt = value; break;
                case "-m": mode = value; break;
                case "-y": system_prompt = value; break;
                case "-r": enable_thinking = Integer.parseInt(value) == 1; break;
                default: error_usage(); break;
            }
        }

        if (rng_seed <= 0) rng_seed = System.currentTimeMillis();
        if (temperature < 0) temperature = 0;
        if (topp < 0 || 1.0 < topp) topp = 0.9f;
        
        try {
            Transformer transformer = new Transformer(checkpoint_path, ctx_length);
            Tokenizer tokenizer = new Tokenizer(checkpoint_path, transformer.config.vocab_size, enable_thinking);
            Sampler sampler = new Sampler(transformer.config.vocab_size, temperature, topp, rng_seed);

            if (prompt == null) {
                System.out.printf("hidden_size=%d, intermediate_size=%d, num_hidden_layers=%d, num_attention_heads=%d, num_kv_heads=%d, head_dim=%d, ctx_length=%d, vocab_size=%d, shared_classifier=%b, quantization_block_size=%d\n", 
                                  transformer.config.dim, transformer.config.hidden_dim, transformer.config.n_layers, transformer.config.n_heads, transformer.config.n_kv_heads, transformer.config.head_dim, transformer.config.seq_len, transformer.config.vocab_size, transformer.config.shared_classifier, transformer.config.group_size);
            }
            
            if ("generate".equals(mode)) {
                generate(transformer, tokenizer, sampler, prompt);
            } else if ("chat".equals(mode)) {
                chat(transformer, tokenizer, sampler, prompt, system_prompt);
            } else {
                System.err.println("Unknown mode: " + mode);
                error_usage();
            }

        } catch (IOException e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}