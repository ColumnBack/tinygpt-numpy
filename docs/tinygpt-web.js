// TinyGPT forward pass in plain JavaScript.
// A line-by-line port of TinyGPT.forward / block_forward / one_head_forward
// in tinygpt.py, used by demo.html. Inference only (no backward).
(function (root) {
  "use strict";

  function decode(b64) {
    var bin = atob(b64);
    var bytes = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    var f32 = new Float32Array(bytes.buffer);
    return Float64Array.from(f32);
  }

  // Y = X @ W (+ b).  X: T x n, W: n x m, row-major
  function matmul(X, T, n, W, m, b) {
    var Y = new Float64Array(T * m);
    for (var t = 0; t < T; t++) {
      for (var j = 0; j < m; j++) {
        var s = b ? b[j] : 0;
        for (var k = 0; k < n; k++) s += X[t * n + k] * W[k * m + j];
        Y[t * m + j] = s;
      }
    }
    return Y;
  }

  function layernorm(R, T, d, gamma, beta, eps) {
    var Y = new Float64Array(T * d);
    for (var t = 0; t < T; t++) {
      var mu = 0, v = 0, k;
      for (k = 0; k < d; k++) mu += R[t * d + k];
      mu /= d;
      for (k = 0; k < d; k++) { var c = R[t * d + k] - mu; v += c * c; }
      var D = Math.sqrt(v / d + eps);
      for (k = 0; k < d; k++) Y[t * d + k] = ((R[t * d + k] - mu) / D) * gamma[k] + beta[k];
    }
    return Y;
  }

  function gelu(x) {
    var c = Math.sqrt(2 / Math.PI);
    return 0.5 * x * (1 + Math.tanh(c * (x + 0.044715 * x * x * x)));
  }

  function softmax(row) {
    var mx = -Infinity, i, s = 0;
    for (i = 0; i < row.length; i++) if (row[i] > mx) mx = row[i];
    var out = new Float64Array(row.length);
    for (i = 0; i < row.length; i++) { out[i] = Math.exp(row[i] - mx); s += out[i]; }
    for (i = 0; i < row.length; i++) out[i] /= s;
    return out;
  }

  function Model(data) {
    var c = data.config;
    this.V = c.vocab_size; this.T = c.max_context; this.d = c.d_model;
    this.H = c.n_heads; this.dh = c.d_model / c.n_heads; this.df = c.d_ff;
    this.N = c.n_layers; this.eps = c.eps;
    this.vocab = data.vocab; this.sentences = data.sentences; this.epoch = data.epoch;
    this.id = {};
    for (var i = 0; i < this.vocab.length; i++) this.id[this.vocab[i]] = i;
    this.p = {};
    for (var k in data.params) this.p[k] = decode(data.params[k].b64);
  }

  // Returns { logits: T x V, attn: [layer][head] -> T x T }
  Model.prototype.forward = function (ids) {
    var T = ids.length, d = this.d, dh = this.dh, p = this.p, t, k, h, n, i, j;
    var X = new Float64Array(T * d);
    for (t = 0; t < T; t++) for (k = 0; k < d; k++) X[t * d + k] = p.E[ids[t] * d + k] + p.P[t * d + k];

    var attn = [];
    for (n = 0; n < this.N; n++) {
      var Ocat = new Float64Array(T * d);
      var layerAttn = [];
      for (h = 0; h < this.H; h++) {
        var Q = matmul(X, T, d, p["WQ_" + n + "_" + h], dh);
        var K = matmul(X, T, d, p["WK_" + n + "_" + h], dh);
        var Vv = matmul(X, T, d, p["WV_" + n + "_" + h], dh);
        var A = new Float64Array(T * T);
        for (i = 0; i < T; i++) {
          var row = new Float64Array(T);
          for (j = 0; j < T; j++) {
            if (j > i) { row[j] = -1e9; continue; }   // causal mask
            var s = 0;
            for (k = 0; k < dh; k++) s += Q[i * dh + k] * K[j * dh + k];
            row[j] = s / Math.sqrt(dh);
          }
          var a = softmax(row);
          for (j = 0; j < T; j++) A[i * T + j] = a[j];
        }
        for (i = 0; i < T; i++) for (k = 0; k < dh; k++) {
          var o = 0;
          for (j = 0; j < T; j++) o += A[i * T + j] * Vv[j * dh + k];
          Ocat[i * d + h * dh + k] = o;
        }
        layerAttn.push(A);
      }
      attn.push(layerAttn);

      var O = matmul(Ocat, T, d, p["WO_" + n], d);
      var R = new Float64Array(T * d);
      for (i = 0; i < T * d; i++) R[i] = X[i] + O[i];
      var Y = layernorm(R, T, d, p["gamma1_" + n], p["beta1_" + n], this.eps);
      var U = matmul(Y, T, d, p["W1_" + n], this.df, p["b1_" + n]);
      for (i = 0; i < U.length; i++) U[i] = gelu(U[i]);
      var F = matmul(U, T, this.df, p["W2_" + n], d, p["b2_" + n]);
      var Z = new Float64Array(T * d);
      for (i = 0; i < T * d; i++) Z[i] = Y[i] + F[i];
      X = layernorm(Z, T, d, p["gamma2_" + n], p["beta2_" + n], this.eps);
    }

    // weight tying: logits = X E^T
    var logits = new Float64Array(T * this.V);
    for (t = 0; t < T; t++) for (var v = 0; v < this.V; v++) {
      var s2 = 0;
      for (k = 0; k < d; k++) s2 += X[t * d + k] * p.E[v * d + k];
      logits[t * this.V + v] = s2;
    }
    return { logits: logits, attn: attn };
  };

  // Next-token distribution after ids, with temperature
  Model.prototype.next = function (ids, temperature) {
    var ctx = ids.slice(-this.T);
    var out = this.forward(ctx);
    var V = this.V, last = ctx.length - 1, temp = Math.max(temperature, 1e-8);
    var row = new Float64Array(V);
    for (var v = 0; v < V; v++) row[v] = out.logits[last * V + v] / temp;
    return { probs: softmax(row), attn: out.attn, context: ctx };
  };

  Model.prototype.sample = function (probs, rand) {
    var r = (rand || Math.random)(), acc = 0;
    for (var i = 0; i < probs.length; i++) { acc += probs[i]; if (r < acc) return i; }
    return probs.length - 1;
  };

  // Same loop as TinyGPT.generate: up to max_context new tokens, stop at <EOS>
  Model.prototype.generate = function (promptIds, temperature, rand) {
    var ids = promptIds.slice(), eos = this.id["<EOS>"];
    for (var s = 0; s < this.T; s++) {
      var nx = this.sample(this.next(ids, temperature).probs, rand);
      ids.push(nx);
      if (nx === eos) break;
    }
    return ids;
  };

  Model.prototype.decode = function (ids) {
    var words = [];
    for (var i = 0; i < ids.length; i++) {
      var tok = this.vocab[ids[i]];
      if (tok === "<BOS>") continue;
      if (tok === "<EOS>") break;
      words.push(tok);
    }
    return words.join(" ");
  };

  root.TinyGPTWeb = { Model: Model };
  if (typeof module !== "undefined") module.exports = root.TinyGPTWeb;
})(typeof window !== "undefined" ? window : globalThis);
