/*

  Reference Cycle Garbage Collection
  ==================================

  Neil Schemenauer <nas@arctrix.com>

  Based on a post on the python-dev list.  Ideas from Guido van Rossum,
  Eric Tiedemann, and various others.

  http://www.arctrix.com/nas/python/gc/

  The following mailing list threads provide a historical perspective on
  the design of this module.  Note that a fair amount of refinement has
  occurred since those discussions.

  http://mail.python.org/pipermail/python-dev/2000-March/002385.html
  http://mail.python.org/pipermail/python-dev/2000-March/002434.html
  http://mail.python.org/pipermail/python-dev/2000-March/002497.html

  For a highlevel view of the collection process, read the collect
  function.

*/

#include "Python.h"
#include "pycore_context.h"
#include "pycore_object.h"
#include "pycore_pymem.h"
#include "pycore_pystate.h"
#include "frameobject.h"        /* for PyFrame_ClearFreeList */
#include "pydtrace.h"
#include "pytime.h"             /* for _PyTime_GetMonotonicClock() */
#include "stdbool.h"

#include "opic/hash/op_hash.h"
#include "opic/hash/op_hash_table.h"

#include <sys/random.h>

/*[clinic input]
module gc
[clinic start generated code]*/
/*[clinic end generated code: output=da39a3ee5e6b4b0d input=b5c9690ecc842d79]*/

#define GC_DEBUG (0)  /* Enable more asserts */

#define GC_NEXT _PyGCHead_NEXT
#define GC_PREV _PyGCHead_PREV

// update_refs() set this bit for all objects in current generation.
// subtract_refs() and move_unreachable() uses this to distinguish
// visited object is in GCing or not.
//
// move_unreachable() removes this flag from reachable objects.
// Only unreachable objects have this flag.
//
// No objects in interpreter have this flag after GC ends.
#define PREV_MASK_COLLECTING   _PyGC_PREV_MASK_COLLECTING

// Lowest bit of _gc_next is used for UNREACHABLE flag.
//
// This flag represents the object is in unreachable list in move_unreachable()
//
// Although this flag is used only in move_unreachable(), move_unreachable()
// doesn't clear this flag to skip unnecessary iteration.
// move_legacy_finalizers() removes this flag instead.
// Between them, unreachable list is not normal list and we can not use
// most gc_list_* functions for it.
#define NEXT_MASK_UNREACHABLE  (1)

/* Get an object's GC head */
#define AS_GC(o) ((PyGC_Head *)(o)-1)

/* Get the object given the GC head */
#define FROM_GC(g) ((PyObject *)(((PyGC_Head *)g)+1))

static inline int
gc_is_collecting(PyGC_Head *g)
{
    return (g->_gc_prev & PREV_MASK_COLLECTING) != 0;
}

static inline void
gc_clear_collecting(PyGC_Head *g)
{
    g->_gc_prev &= ~PREV_MASK_COLLECTING;
}

static inline Py_ssize_t
gc_get_refs(PyGC_Head *g)
{
    return (Py_ssize_t)(g->_gc_prev >> _PyGC_PREV_SHIFT);
}

static inline void
gc_set_refs(PyGC_Head *g, Py_ssize_t refs)
{
    g->_gc_prev = (g->_gc_prev & ~_PyGC_PREV_MASK)
        | ((uintptr_t)(refs) << _PyGC_PREV_SHIFT);
}

static inline void
gc_reset_refs(PyGC_Head *g, Py_ssize_t refs)
{
    g->_gc_prev = (g->_gc_prev & _PyGC_PREV_MASK_FINALIZED)
        | PREV_MASK_COLLECTING
        | ((uintptr_t)(refs) << _PyGC_PREV_SHIFT);
}

static inline void
gc_decref(PyGC_Head *g)
{
    _PyObject_ASSERT_WITH_MSG(FROM_GC(g),
                              gc_get_refs(g) > 0,
                              "refcount is too small");
    g->_gc_prev -= 1 << _PyGC_PREV_SHIFT;
}

/* Python string to use if unhandled exception occurs */
static PyObject *gc_str = NULL;

/* set for debugging information */
#define DEBUG_STATS             (1<<0) /* print collection statistics */
#define DEBUG_COLLECTABLE       (1<<1) /* print collectable objects */
#define DEBUG_UNCOLLECTABLE     (1<<2) /* print uncollectable objects */
#define DEBUG_SAVEALL           (1<<5) /* save all garbage in gc.garbage */
#define DEBUG_LEAK              DEBUG_COLLECTABLE | \
                DEBUG_UNCOLLECTABLE | \
                DEBUG_SAVEALL

#define GEN_HEAD(state, n) (&(state)->generations[n].head)

// Memory usage for learning GC. This value is written by `obmalloc.c`.
size_t memory_usage = 0;

// Q configuration.
#define Q_NUM_ACTIONS 4

// Struct for Q observation.
typedef struct QObservation {
    // Memory usage in bytes.
    uint64_t memory;
    // Instruction location.
    uint64_t instruction;
} QObservation;

// Struct for Q transition to be stored in replay memory. Usually, we would use a sequence here. However, we simply
// use 1 observation.
typedef struct QTransition {
    // The state before taking the action.
    QObservation observation;
    // The reward associated with the transition.
    double reward;
    // The action taken.
    uint8_t action;
} QTransition;

// Struct for Q config.
static struct QConfig {
    // Capacity for replay memory in bytes. Must be a power of 2.
    uint64_t replay_size;
    // Learning rate for ANN.
    double learning_rate;
    // Epsilon values for greedy strategy.
    double epsilon_start;
    double epsilon_end;
    double epsilon_decay;
    // Gamma value for non-terminal sequences.
    double gamma;
    // The number of objects to skip before performing evaluation. Must be a power of 2.
    uint64_t skip;
    // The maximum memory in bytes for Q-table.
    uint64_t max_memory;
    // The fence memory and penalty.
    uint64_t fence_memory;
    double fence_penalty;
    // The memory step size.
    uint16_t memory_step;
} q_config;

// Struct for Q state.
static struct QState {
    // The sparse table representing Q-values.
    OPHashTable *q_table;
    // Replay memory.
    QTransition *replay;
    // The number of entries in the replay memory.
    uint64_t replay_capacity;
    // A circular index into the replay memory.
    uint64_t replay_index;
    // The last index that was sampled from. At most the replay index.
    uint64_t last_replay_index;
    // The number of evaluations.
    uint64_t evaluations;
    // The number of objects seen.
    uint64_t objects;
    // The number of random actions taken.
    uint64_t random_actions;
    // Whether the learning GC is enabled.
    bool enabled;
    // Whether we've issued a warning about the replay table.
    bool did_warn;
    // The training thread ID.
    pthread_t tid;
    // Argument to training thread.
    double arg_reward;
    // Whether the replay table is locked.
    bool replay_locked;
} q_state;

// Key for Q-table.
typedef struct QKey {
    // The packed value including the instruction and the memory.
    uint64_t packed;
} QKey;

/*** Begin Q functions. ***/

// Rounds a number to the next power of 2.
static uint64_t q_next_power_of_two(uint64_t v) {
    v--;
    v |= v >> 1U;
    v |= v >> 2U;
    v |= v >> 4U;
    v |= v >> 8U;
    v |= v >> 16U;
    v |= v >> 32U;
    v++;
    return v;
}

// Gets the index in a circular buffer.
static inline uint64_t q_index(uint64_t index) {
    return index & (q_state.replay_capacity - 1);
}

// Creates a Q-table key.
static QKey q_create_key(uint64_t instruction, uint64_t memory) {
    // Memory should be capped.
    if (memory > q_config.max_memory) {
        memory = q_config.max_memory;
    }

    // Quantize memory.
    memory >>= q_config.memory_step;

    // Pack instruction and memory.
    return (QKey) {.packed = (instruction << 16U) | memory};
}

// Gets the instruction of a Q-table key.
static inline uint64_t q_key_instruction(QKey key) {
    return key.packed >> 16U;
}

// Gets the memory of a Q-table key.
static inline uint64_t q_key_memory(QKey key) {
    return key.packed & 0xFFFFU;
}

// Hashes a key using the Thomas Wang's 64-bit mix function.
static inline uint64_t q_hash(void *key_generic, size_t size) {
    uint64_t key = ((QKey *) key_generic)->packed;
    key = (~key) + (key << 21U);
    key = key ^ (key >> 24U);
    key = (key + (key << 3U)) + (key << 8U);
    key = key ^ (key >> 14U);
    key = (key + (key << 2U)) + (key << 4U);
    key = key ^ (key >> 28U);
    key = key + (key << 31U);
    return key;
}

// Seed for random number generator.
static uint64_t q_seed[4];

// The rotl function used for xoshiro256++.
static inline uint64_t q_rotl(const uint64_t x, unsigned int k) {
    return (x << k) | (x >> (64U - k));
}

// Generate a random 64-bit integer using xoshiro256++.
static uint64_t q_rand() {
    const uint64_t result = q_rotl(q_seed[0] + q_seed[3], 23) + q_seed[0];

    q_seed[2] ^= q_seed[0];
    q_seed[3] ^= q_seed[1];
    q_seed[1] ^= q_seed[2];
    q_seed[0] ^= q_seed[3];
    q_seed[2] ^= q_seed[1] << 17U;
    q_seed[3] = q_rotl(q_seed[3], 45U);

    return result;
}

// Generate a 64-bit integer within a given range of [min, max).
static uint64_t q_rand_range(uint64_t min, uint64_t max) __attribute__((unused)) {
    return min + (q_rand() % (max - min));
}

// Seed the random number generator.
void q_seed_rand() {
    // Use a syscall.
    getrandom(&q_seed, sizeof(q_seed), 0);
}

// Untag pointer.
static inline uintptr_t q_untag(uintptr_t ptr) {
    return ptr & ~0x7ULL;
}

// Tag pointer.
static inline uintptr_t q_tag(uintptr_t ptr, uintptr_t tag) {
    return q_untag(ptr) | (tag & 0x7ULL);
}

// Takes an observation.
static QObservation q_observation() {
    QObservation observation;

    // The memory is just the current memory.
    observation.memory = memory_usage;

    // Update the instruction location from the thread state.
    // TODO(bobbyluig): Find a faster way to get the instruction.
    PyThreadState *ts = PyThreadState_Get();
    if (ts->frame != NULL && ts->frame->f_lasti >= 0) {
        observation.instruction = ts->frame->f_code->co_id + ts->frame->f_lasti / sizeof(_Py_CODEUNIT);
    } else {
        observation.instruction = 0;
    }

    return observation;
}

// Gets a pointer to the Q-value table associated with a replay index, creating the table as necessary if it does not
// exist and populates it with the correct initial values. This method should only be called by the training thread and
// not by any evaluation threads.
static uintptr_t *q_get_table_ptr(uint64_t index) {
    // Fetch the key.
    QObservation observation = q_state.replay[index].observation;
    QKey key = q_create_key(observation.instruction, observation.memory);

    // It is more efficient to use an upsert here, but we have to get and then upsert because insertion requires a lock
    // on the hashtable. The hope is that we don't have to do this very often, and once the hashtable is initialized,
    // there is not really a need to acquire and release locks.
    uintptr_t *table_ptr = HTGetCustom(q_state.q_table, q_hash, &key);

    // If the key does exist, we are done.
    if (table_ptr != NULL) {
        return table_ptr;
    }

    // Allocate a new table. All values are initially zero.
    double *table = PyMem_RawCalloc(Q_NUM_ACTIONS, sizeof(double));

    // Define the tag (best policy index).
    uint8_t tag = 0;

    // If over memory fence, penalize every state except full collection.
    if (observation.memory > q_config.fence_memory) {
        // Apply penalty.
        for (int i = 0; i < Q_NUM_ACTIONS - 1; i++) {
            table[i] -= q_config.fence_penalty;
        }

        // The best index is to do full collection if there is any penalty.
        if (q_config.fence_penalty > 0) {
            tag = Q_NUM_ACTIONS - 1;
        }
    }

    // Acquire GIL to perform upsert.
    PyGILState_STATE gstate = PyGILState_Ensure();

    // Do upsert, ignoring duplicate flag.
    bool is_duplicate;
    HTUpsertCustom(q_state.q_table, q_hash, &key, (void **) &table_ptr, &is_duplicate);
    *table_ptr = q_tag((uintptr_t) table, tag);

    // Release GIL whe insertion is complete.
    PyGILState_Release(gstate);

    // Return the pointer to the table.
    return table_ptr;
}

// Pushes a frame into replay memory and returns the address of the replay.
static QTransition *q_replay_push(QObservation *observation) {
    // Create transition.
    QTransition transition;
    transition.observation = *observation;
    transition.reward = 0;
    transition.action = 0;

    // Store into replay memory.
    uint64_t index = q_index(q_state.replay_index);
    q_state.replay[index] = transition;

    // If the replay table is not locked, increment the replay index.
    q_state.replay_index += !q_state.replay_locked;

    // Return the address of the replay function.
    return &q_state.replay[index];
}

// Gets the last frame in the replay memory. This method should only be called after a call to push an observation.
static QTransition *q_last_replay() {
    // When the replay is locked, then the training thread is running. The replay index does not change, and therefore
    // the previous index is just the replay index.
    uint64_t previous_index = q_index(q_state.replay_index - !q_state.replay_locked);
    return &(q_state.replay[previous_index]);
}

// Finds the index of the action associated with the highest value.
static uint8_t q_max_index(const double *table) {
    // Find the best index.
    uint8_t best_index = 0;
    for (int i = 1; i < Q_NUM_ACTIONS; i++) {
        if (table[i] > table[best_index]) {
            best_index = i;
        }
    }
    return best_index;
}

// Evaluates the policy function and returns the index of the best output. In the case that there are no entries in the
// Q-table corresponding to the state, then the index of the best value under the default policy is returned.
static uint8_t q_evaluate(QTransition *replay) {
    // Fetch the entry from the hashtable.
    QObservation observation = replay->observation;
    QKey key = q_create_key(observation.instruction, observation.memory);
    uintptr_t *table_ptr = HTGetCustom(q_state.q_table, q_hash, &key);

    // If the value exists, extract from the bottom two bits.
    if (table_ptr != NULL) {
        return *table_ptr & 0x3U;
    }

    // The value doesn't exist. Therefore, the default policy is to do full collection if we are above the memory
    // threshold. Otherwise, do no collection.
    if (observation.memory > q_config.fence_memory) {
        return Q_NUM_ACTIONS - 1;
    } else {
        return 0;
    }
}

// Thresholds for random actions.
#define THRESHOLD_0 (10.0)
#define THRESHOLD_1 (10.0)
#define THRESHOLD_2 (10.0)

// Do not change these below.
#define THRESHOLD_DENOMINATOR (1.0 + THRESHOLD_2 \
                                   + THRESHOLD_2 * THRESHOLD_1 \
                                   + THRESHOLD_2 * THRESHOLD_1 * THRESHOLD_0)
#define THRESHOLD_0_VAL ((THRESHOLD_2 * THRESHOLD_1 * THRESHOLD_0) / THRESHOLD_DENOMINATOR)
#define THRESHOLD_1_VAL (THRESHOLD_0_VAL + (THRESHOLD_2 * THRESHOLD_1) / THRESHOLD_DENOMINATOR)
#define THRESHOLD_2_VAL (THRESHOLD_1_VAL + (THRESHOLD_2) / THRESHOLD_DENOMINATOR)

// Selects a random action from the prior distribution.
inline static uint8_t q_random_action() {
    // Indicate that a random action was chosen.
    q_state.random_actions++;

    // Select from prior distribution.
    uint64_t r = q_rand();
    if (r < THRESHOLD_0_VAL * UINT64_MAX) {
        return 0;
    } else if (r < THRESHOLD_1_VAL * UINT64_MAX) {
        return 1;
    } else if (r < THRESHOLD_2_VAL * UINT64_MAX) {
        return 2;
    } else {
        return 3;
    }
}

// Determines whether the epsilon greedy strategy should be used.
static inline bool q_use_random_action() {
    // Compute threshold.
    double eps_threshold = q_config.epsilon_end + (q_config.epsilon_start - q_config.epsilon_end)
                                                  * exp(-1.0 * q_state.evaluations / q_config.epsilon_decay);

    // Determine if we should just use the policy action instead of a random action.
    return (q_rand() <= eps_threshold * UINT64_MAX);
}

// Clears randomization for a hashtable entry.
static void q_clear_randomization_entry(void *key_generic, void *value_generic,
                                        size_t keysize, size_t valsize, void *context) {
    // Get the pointer to the table.
    uintptr_t *value = (uintptr_t *) value_generic;

    // If the tag bit is set, then the entry was force randomized. Reset it.
    if (*value & 0x4U) {
        // Set the max value.
        double *table = (double *) q_untag(*value);
        *value = q_tag(*value, q_max_index(table));
    }
}

// Clears randomization.
static void q_clear_randomization() __attribute__((unused)) {
    // Apply to all entries.
    HTIterate(q_state.q_table, q_clear_randomization_entry, NULL);
}

// Trains on a given index in the replay table.
static void q_train_index(uint64_t index) {
    // Extract values.
    uint8_t action = q_state.replay[index].action;
    double reward = q_state.replay[index].reward;

    // Get this table.
    uintptr_t *table_ptr = q_get_table_ptr(index);
    double *table = (double *) q_untag(*table_ptr);

    // Get the future table.
    uintptr_t *future_table_ptr = q_get_table_ptr(q_index(index + 1));
    double *future_table = (double *) q_untag(*future_table_ptr);

    // Estimate the optimal future value. Note that we do not use the cached values here, since those could have come
    // from the randomized epsilon-greedy strategy.
    uint8_t future_action = q_max_index(future_table);
    double q_future = future_table[future_action];

    // Compute desired quality value.
    double old_y = table[action];
    double y = old_y + q_config.learning_rate * (reward + q_config.gamma * q_future - old_y);

    // Update Q-value.
    table[action] = y;

    // Determine the best action and set the tag bits.
    *table_ptr = q_tag((uintptr_t) table, q_max_index(table));
}

// Frees a hashtable entry.
static void q_free_hashtable_entry(void *key_generic, void *value_generic,
                                   size_t keysize, size_t valsize, void *context) {
    // Free value.
    uintptr_t *value = (uintptr_t *) value_generic;
    PyMem_RawFree((void *) q_untag(*value));
}

// Frees a hashtable entry.
static void q_print_hashtable_entry(void *key_generic, void *value_generic,
                                    size_t keysize, size_t valsize, void *context) {
    // Get the key and value.
    QKey *key = (QKey *) key_generic;
    uintptr_t *value = (uintptr_t *) value_generic;

    // Find the maximum value.
    uint8_t action = q_max_index((double *) q_untag(*value));

    // Print key and value if the action is to collect.
    if (action > 0) {
        printf("(%lu, %lu): %u\n", q_key_instruction(*key), q_key_memory(*key), action);
    }
}

// Method to determine which generation to collect, or -1 for no collection.
static int _Py_HOT_FUNCTION
q_predict(struct _gc_runtime_state *state) {
    // Determine whether we should use Q. Otherwise, use the default GC.
    if (q_state.enabled) {
        // Determine whether it is time to use the policy function.
        if ((q_state.objects++ & (q_config.skip - 1)) != 0) {
            return -1;
        }

        // Store the observation and select an action.
        QObservation observation = q_observation();
        QTransition *replay = q_replay_push(&observation);

        // Determine if we should evaluate of use the epsilon-greedy strategy.
        uint8_t action = q_use_random_action()
                ? q_random_action()
                : q_evaluate(replay);

        // Store the chosen action.
        replay->action = action;

        // Convert to GC action.
        return ((int) action) - 1;
    }

    // If the count of generation 0 does not exceed the threshold, do nothing.
    if (state->generations[0].count <= state->generations[0].threshold) {
        return -1;
    }

    // Determine which generation to collect.
    for (int i = NUM_GENERATIONS - 1; i >= 1; i--) {
        if (state->generations[i].count > state->generations[i].threshold) {
            if (i == NUM_GENERATIONS - 1 && state->long_lived_pending < state->long_lived_total / 4) {
                continue;
            }
            return i;
        }
    }

    // The first generation should be collected.
    return 0;
}

// Performs training for a given reward.
static void *q_train(void *_) {
    // Get the reward.
    double reward = q_state.arg_reward;

    // Check whether we exceeded the size of the replay table.
    if (q_state.replay_index - q_state.last_replay_index >= q_state.replay_capacity) {
        // Warn about exceeding the replay table size.
        if (!q_state.did_warn) {
            q_state.did_warn = true;
            printf("Please increase the size of the replay table.\n");
        }

        // Apply reward to everything in the replay table.
        for (uint64_t i = 0; i < q_state.replay_capacity; i++) {
            q_state.replay[i].reward += reward;
        }

        // Do training on everything in the replay table backwards.
        for (uint64_t i = q_state.replay_index; i != q_state.replay_index - q_state.replay_capacity; i--) {
            q_train_index(q_index(i));
        }
    } else {
        // Apply reward to new observations.
        for (uint64_t i = q_state.last_replay_index; i < q_state.replay_index; i++) {
            q_state.replay[q_index(i)].reward += reward;
        }

        // Try to find what the ending index is. We want to train on one before the last replay index if possible.
        uint64_t end = (q_state.last_replay_index == 0)
                       ? q_state.last_replay_index - 1
                       : q_state.last_replay_index - 2;

        // Do some training. The replay index is one after the last entry. We start at two before the replay index,
        // which is the second to last entry and therefore has a next entry for future value computation.
        for (uint64_t i = q_state.replay_index - 2; i != end; i--) {
            q_train_index(q_index(i));
        }
    }

    // Set last replay index.
    q_state.last_replay_index = q_state.replay_index;

    // Increase the number of evaluations for epsilon-greedy strategy.
    q_state.evaluations++;

    // It's tempting to acquire GIL here before unlocking the table so that we are not in the middle of instruction.
    // However, we don't need to do. This increase performance at the expense of the loss of bit of of reward shaping
    // information if this somehow happens in the middle of an GC allocation.
    q_state.replay_locked = false;

    // Return nothing.
    return NULL;
}

/*** End Q functions. ***/

void
_PyGC_Initialize(struct _gc_runtime_state *state)
{
    state->enabled = 1; /* automatic collection enabled? */

#define _GEN_HEAD(n) GEN_HEAD(state, n)
    struct gc_generation generations[NUM_GENERATIONS] = {
        /* PyGC_Head,                                    threshold,    count */
        {{(uintptr_t)_GEN_HEAD(0), (uintptr_t)_GEN_HEAD(0)},   700,        0},
        {{(uintptr_t)_GEN_HEAD(1), (uintptr_t)_GEN_HEAD(1)},   10,         0},
        {{(uintptr_t)_GEN_HEAD(2), (uintptr_t)_GEN_HEAD(2)},   10,         0},
    };
    for (int i = 0; i < NUM_GENERATIONS; i++) {
        state->generations[i] = generations[i];
    };
    state->generation0 = GEN_HEAD(state, 0);
    struct gc_generation permanent_generation = {
          {(uintptr_t)&state->permanent_generation.head,
           (uintptr_t)&state->permanent_generation.head}, 0, 0
    };
    state->permanent_generation = permanent_generation;

    // Parse configuration parameters.
    const char *q_replay_size = Py_GETENV("Q_REPLAY_SIZE");
    const char *q_learning_rate = Py_GETENV("Q_LEARNING_RATE");
    const char *q_epsilon_start = Py_GETENV("Q_EPSILON_START");
    const char *q_epsilon_end = Py_GETENV("Q_EPSILON_END");
    const char *q_epsilon_decay = Py_GETENV("Q_EPSILON_DECAY");
    const char *q_gamma = Py_GETENV("Q_GAMMA");
    const char *q_skip = Py_GETENV("Q_SKIP");
    const char *q_max_memory = Py_GETENV("Q_MAX_MEMORY");
    const char *q_fence_memory = Py_GETENV("Q_FENCE_MEMORY");
    const char *q_fence_penalty = Py_GETENV("Q_FENCE_PENALTY");
    const char *q_memory_step = Py_GETENV("Q_MEMORY_STEP");
    q_config.replay_size = (q_replay_size != NULL) ? (uint64_t) atoll(q_replay_size) : (16ULL << 20U);
    q_config.learning_rate = (q_learning_rate != NULL) ? atof(q_learning_rate) : 0.1;
    q_config.epsilon_start = (q_epsilon_start != NULL) ? atof(q_epsilon_start) : 1.0;
    q_config.epsilon_end = (q_epsilon_end != NULL) ? atof(q_epsilon_end) : 0.01;
    q_config.epsilon_decay = (q_epsilon_decay != NULL) ? atof(q_epsilon_decay) : 10.0;
    q_config.gamma = (q_gamma != NULL) ? atof(q_gamma) : 0.9999;
    q_config.skip = (q_skip != NULL) ? (uint64_t) atoll(q_skip) : 16ULL;
    q_config.max_memory = (q_max_memory != NULL) ? (uint64_t) atoll(q_max_memory) : (256ULL << 20U);
    q_config.fence_memory = (q_fence_memory != NULL) ? (uint64_t) atoll(q_fence_memory) : q_config.max_memory;
    q_config.fence_penalty = (q_fence_penalty != NULL) ? atof(q_fence_penalty) : 1.0;
    q_config.memory_step = (q_memory_step != NULL) ? (uint16_t) atoi(q_memory_step) : 20U;

    // Round to next power of 2 for some parameters.
    q_config.replay_size = q_next_power_of_two(q_config.replay_size);
    q_config.skip = q_next_power_of_two(q_config.skip);

    // Seed RNG.
    q_seed_rand();

    // Initialize Q state.
    q_state.q_table = HTNew(1024, 0.89, sizeof(QKey), sizeof(uintptr_t));
    q_state.replay_capacity = q_config.replay_size / sizeof(QTransition);
    q_state.replay_index = 0;
    q_state.last_replay_index = 0;
    q_state.replay = PyMem_RawMalloc(q_state.replay_capacity * sizeof(QTransition));
    q_state.evaluations = 0;
    q_state.objects = 0;
    q_state.random_actions = 0;
    q_state.enabled = false;
    q_state.did_warn = false;
    q_state.replay_locked = false;
}

/*
_gc_prev values
---------------

Between collections, _gc_prev is used for doubly linked list.

Lowest two bits of _gc_prev are used for flags.
PREV_MASK_COLLECTING is used only while collecting and cleared before GC ends
or _PyObject_GC_UNTRACK() is called.

During a collection, _gc_prev is temporary used for gc_refs, and the gc list
is singly linked until _gc_prev is restored.

gc_refs
    At the start of a collection, update_refs() copies the true refcount
    to gc_refs, for each object in the generation being collected.
    subtract_refs() then adjusts gc_refs so that it equals the number of
    times an object is referenced directly from outside the generation
    being collected.

PREV_MASK_COLLECTING
    Objects in generation being collected are marked PREV_MASK_COLLECTING in
    update_refs().


_gc_next values
---------------

_gc_next takes these values:

0
    The object is not tracked

!= 0
    Pointer to the next object in the GC list.
    Additionally, lowest bit is used temporary for
    NEXT_MASK_UNREACHABLE flag described below.

NEXT_MASK_UNREACHABLE
    move_unreachable() then moves objects not reachable (whether directly or
    indirectly) from outside the generation into an "unreachable" set and
    set this flag.

    Objects that are found to be reachable have gc_refs set to 1.
    When this flag is set for the reachable object, the object must be in
    "unreachable" set.
    The flag is unset and the object is moved back to "reachable" set.

    move_legacy_finalizers() will remove this flag from "unreachable" set.
*/

/*** list functions ***/

static inline void
gc_list_init(PyGC_Head *list)
{
    // List header must not have flags.
    // We can assign pointer by simple cast.
    list->_gc_prev = (uintptr_t)list;
    list->_gc_next = (uintptr_t)list;
}

static inline int
gc_list_is_empty(PyGC_Head *list)
{
    return (list->_gc_next == (uintptr_t)list);
}

/* Append `node` to `list`. */
static inline void
gc_list_append(PyGC_Head *node, PyGC_Head *list)
{
    PyGC_Head *last = (PyGC_Head *)list->_gc_prev;

    // last <-> node
    _PyGCHead_SET_PREV(node, last);
    _PyGCHead_SET_NEXT(last, node);

    // node <-> list
    _PyGCHead_SET_NEXT(node, list);
    list->_gc_prev = (uintptr_t)node;
}

/* Remove `node` from the gc list it's currently in. */
static inline void
gc_list_remove(PyGC_Head *node)
{
    PyGC_Head *prev = GC_PREV(node);
    PyGC_Head *next = GC_NEXT(node);

    _PyGCHead_SET_NEXT(prev, next);
    _PyGCHead_SET_PREV(next, prev);

    node->_gc_next = 0; /* object is not currently tracked */
}

/* Move `node` from the gc list it's currently in (which is not explicitly
 * named here) to the end of `list`.  This is semantically the same as
 * gc_list_remove(node) followed by gc_list_append(node, list).
 */
static void
gc_list_move(PyGC_Head *node, PyGC_Head *list)
{
    /* Unlink from current list. */
    PyGC_Head *from_prev = GC_PREV(node);
    PyGC_Head *from_next = GC_NEXT(node);
    _PyGCHead_SET_NEXT(from_prev, from_next);
    _PyGCHead_SET_PREV(from_next, from_prev);

    /* Relink at end of new list. */
    // list must not have flags.  So we can skip macros.
    PyGC_Head *to_prev = (PyGC_Head*)list->_gc_prev;
    _PyGCHead_SET_PREV(node, to_prev);
    _PyGCHead_SET_NEXT(to_prev, node);
    list->_gc_prev = (uintptr_t)node;
    _PyGCHead_SET_NEXT(node, list);
}

/* append list `from` onto list `to`; `from` becomes an empty list */
static void
gc_list_merge(PyGC_Head *from, PyGC_Head *to)
{
    assert(from != to);
    if (!gc_list_is_empty(from)) {
        PyGC_Head *to_tail = GC_PREV(to);
        PyGC_Head *from_head = GC_NEXT(from);
        PyGC_Head *from_tail = GC_PREV(from);
        assert(from_head != from);
        assert(from_tail != from);

        _PyGCHead_SET_NEXT(to_tail, from_head);
        _PyGCHead_SET_PREV(from_head, to_tail);

        _PyGCHead_SET_NEXT(from_tail, to);
        _PyGCHead_SET_PREV(to, from_tail);
    }
    gc_list_init(from);
}

static Py_ssize_t
gc_list_size(PyGC_Head *list)
{
    PyGC_Head *gc;
    Py_ssize_t n = 0;
    for (gc = GC_NEXT(list); gc != list; gc = GC_NEXT(gc)) {
        n++;
    }
    return n;
}

/* Append objects in a GC list to a Python list.
 * Return 0 if all OK, < 0 if error (out of memory for list).
 */
static int
append_objects(PyObject *py_list, PyGC_Head *gc_list)
{
    PyGC_Head *gc;
    for (gc = GC_NEXT(gc_list); gc != gc_list; gc = GC_NEXT(gc)) {
        PyObject *op = FROM_GC(gc);
        if (op != py_list) {
            if (PyList_Append(py_list, op)) {
                return -1; /* exception */
            }
        }
    }
    return 0;
}

#if GC_DEBUG
// validate_list checks list consistency.  And it works as document
// describing when expected_mask is set / unset.
static void
validate_list(PyGC_Head *head, uintptr_t expected_mask)
{
    PyGC_Head *prev = head;
    PyGC_Head *gc = GC_NEXT(head);
    while (gc != head) {
        assert(GC_NEXT(gc) != NULL);
        assert(GC_PREV(gc) == prev);
        assert((gc->_gc_prev & PREV_MASK_COLLECTING) == expected_mask);
        prev = gc;
        gc = GC_NEXT(gc);
    }
    assert(prev == GC_PREV(head));
}
#else
#define validate_list(x,y) do{}while(0)
#endif

/*** end of list stuff ***/


/* Set all gc_refs = ob_refcnt.  After this, gc_refs is > 0 and
 * PREV_MASK_COLLECTING bit is set for all objects in containers.
 */
static void
update_refs(PyGC_Head *containers)
{
    PyGC_Head *gc = GC_NEXT(containers);
    for (; gc != containers; gc = GC_NEXT(gc)) {
        gc_reset_refs(gc, Py_REFCNT(FROM_GC(gc)));
        /* Python's cyclic gc should never see an incoming refcount
         * of 0:  if something decref'ed to 0, it should have been
         * deallocated immediately at that time.
         * Possible cause (if the assert triggers):  a tp_dealloc
         * routine left a gc-aware object tracked during its teardown
         * phase, and did something-- or allowed something to happen --
         * that called back into Python.  gc can trigger then, and may
         * see the still-tracked dying object.  Before this assert
         * was added, such mistakes went on to allow gc to try to
         * delete the object again.  In a debug build, that caused
         * a mysterious segfault, when _Py_ForgetReference tried
         * to remove the object from the doubly-linked list of all
         * objects a second time.  In a release build, an actual
         * double deallocation occurred, which leads to corruption
         * of the allocator's internal bookkeeping pointers.  That's
         * so serious that maybe this should be a release-build
         * check instead of an assert?
         */
        _PyObject_ASSERT(FROM_GC(gc), gc_get_refs(gc) != 0);
    }
}

/* A traversal callback for subtract_refs. */
static int
visit_decref(PyObject *op, void *parent)
{
    _PyObject_ASSERT(_PyObject_CAST(parent), !_PyObject_IsFreed(op));

    if (PyObject_IS_GC(op)) {
        PyGC_Head *gc = AS_GC(op);
        /* We're only interested in gc_refs for objects in the
         * generation being collected, which can be recognized
         * because only they have positive gc_refs.
         */
        if (gc_is_collecting(gc)) {
            gc_decref(gc);
        }
    }
    return 0;
}

/* Subtract internal references from gc_refs.  After this, gc_refs is >= 0
 * for all objects in containers, and is GC_REACHABLE for all tracked gc
 * objects not in containers.  The ones with gc_refs > 0 are directly
 * reachable from outside containers, and so can't be collected.
 */
static void
subtract_refs(PyGC_Head *containers)
{
    traverseproc traverse;
    PyGC_Head *gc = GC_NEXT(containers);
    for (; gc != containers; gc = GC_NEXT(gc)) {
        PyObject *op = FROM_GC(gc);
        traverse = Py_TYPE(op)->tp_traverse;
        (void) traverse(FROM_GC(gc),
                       (visitproc)visit_decref,
                       op);
    }
}

/* A traversal callback for move_unreachable. */
static int
visit_reachable(PyObject *op, PyGC_Head *reachable)
{
    if (!PyObject_IS_GC(op)) {
        return 0;
    }

    PyGC_Head *gc = AS_GC(op);
    const Py_ssize_t gc_refs = gc_get_refs(gc);

    // Ignore untracked objects and objects in other generation.
    if (gc->_gc_next == 0 || !gc_is_collecting(gc)) {
        return 0;
    }

    if (gc->_gc_next & NEXT_MASK_UNREACHABLE) {
        /* This had gc_refs = 0 when move_unreachable got
         * to it, but turns out it's reachable after all.
         * Move it back to move_unreachable's 'young' list,
         * and move_unreachable will eventually get to it
         * again.
         */
        // Manually unlink gc from unreachable list because
        PyGC_Head *prev = GC_PREV(gc);
        PyGC_Head *next = (PyGC_Head*)(gc->_gc_next & ~NEXT_MASK_UNREACHABLE);
        _PyObject_ASSERT(FROM_GC(prev),
                         prev->_gc_next & NEXT_MASK_UNREACHABLE);
        _PyObject_ASSERT(FROM_GC(next),
                         next->_gc_next & NEXT_MASK_UNREACHABLE);
        prev->_gc_next = gc->_gc_next;  // copy NEXT_MASK_UNREACHABLE
        _PyGCHead_SET_PREV(next, prev);

        gc_list_append(gc, reachable);
        gc_set_refs(gc, 1);
    }
    else if (gc_refs == 0) {
        /* This is in move_unreachable's 'young' list, but
         * the traversal hasn't yet gotten to it.  All
         * we need to do is tell move_unreachable that it's
         * reachable.
         */
        gc_set_refs(gc, 1);
    }
    /* Else there's nothing to do.
     * If gc_refs > 0, it must be in move_unreachable's 'young'
     * list, and move_unreachable will eventually get to it.
     */
    else {
        _PyObject_ASSERT_WITH_MSG(op, gc_refs > 0, "refcount is too small");
    }
    return 0;
}

/* Move the unreachable objects from young to unreachable.  After this,
 * all objects in young don't have PREV_MASK_COLLECTING flag and
 * unreachable have the flag.
 * All objects in young after this are directly or indirectly reachable
 * from outside the original young; and all objects in unreachable are
 * not.
 *
 * This function restores _gc_prev pointer.  young and unreachable are
 * doubly linked list after this function.
 * But _gc_next in unreachable list has NEXT_MASK_UNREACHABLE flag.
 * So we can not gc_list_* functions for unreachable until we remove the flag.
 */
static void
move_unreachable(PyGC_Head *young, PyGC_Head *unreachable)
{
    // previous elem in the young list, used for restore gc_prev.
    PyGC_Head *prev = young;
    PyGC_Head *gc = GC_NEXT(young);

    /* Invariants:  all objects "to the left" of us in young are reachable
     * (directly or indirectly) from outside the young list as it was at entry.
     *
     * All other objects from the original young "to the left" of us are in
     * unreachable now, and have NEXT_MASK_UNREACHABLE.  All objects to the
     * left of us in 'young' now have been scanned, and no objects here
     * or to the right have been scanned yet.
     */

    while (gc != young) {
        if (gc_get_refs(gc)) {
            /* gc is definitely reachable from outside the
             * original 'young'.  Mark it as such, and traverse
             * its pointers to find any other objects that may
             * be directly reachable from it.  Note that the
             * call to tp_traverse may append objects to young,
             * so we have to wait until it returns to determine
             * the next object to visit.
             */
            PyObject *op = FROM_GC(gc);
            traverseproc traverse = Py_TYPE(op)->tp_traverse;
            _PyObject_ASSERT_WITH_MSG(op, gc_get_refs(gc) > 0,
                                      "refcount is too small");
            // NOTE: visit_reachable may change gc->_gc_next when
            // young->_gc_prev == gc.  Don't do gc = GC_NEXT(gc) before!
            (void) traverse(op,
                    (visitproc)visit_reachable,
                    (void *)young);
            // relink gc_prev to prev element.
            _PyGCHead_SET_PREV(gc, prev);
            // gc is not COLLECTING state after here.
            gc_clear_collecting(gc);
            prev = gc;
        }
        else {
            /* This *may* be unreachable.  To make progress,
             * assume it is.  gc isn't directly reachable from
             * any object we've already traversed, but may be
             * reachable from an object we haven't gotten to yet.
             * visit_reachable will eventually move gc back into
             * young if that's so, and we'll see it again.
             */
            // Move gc to unreachable.
            // No need to gc->next->prev = prev because it is single linked.
            prev->_gc_next = gc->_gc_next;

            // We can't use gc_list_append() here because we use
            // NEXT_MASK_UNREACHABLE here.
            PyGC_Head *last = GC_PREV(unreachable);
            // NOTE: Since all objects in unreachable set has
            // NEXT_MASK_UNREACHABLE flag, we set it unconditionally.
            // But this may set the flat to unreachable too.
            // move_legacy_finalizers() should care about it.
            last->_gc_next = (NEXT_MASK_UNREACHABLE | (uintptr_t)gc);
            _PyGCHead_SET_PREV(gc, last);
            gc->_gc_next = (NEXT_MASK_UNREACHABLE | (uintptr_t)unreachable);
            unreachable->_gc_prev = (uintptr_t)gc;
        }
        gc = (PyGC_Head*)prev->_gc_next;
    }
    // young->_gc_prev must be last element remained in the list.
    young->_gc_prev = (uintptr_t)prev;
}

static void
untrack_tuples(PyGC_Head *head)
{
    PyGC_Head *next, *gc = GC_NEXT(head);
    while (gc != head) {
        PyObject *op = FROM_GC(gc);
        next = GC_NEXT(gc);
        if (PyTuple_CheckExact(op)) {
            _PyTuple_MaybeUntrack(op);
        }
        gc = next;
    }
}

/* Try to untrack all currently tracked dictionaries */
static void
untrack_dicts(PyGC_Head *head)
{
    PyGC_Head *next, *gc = GC_NEXT(head);
    while (gc != head) {
        PyObject *op = FROM_GC(gc);
        next = GC_NEXT(gc);
        if (PyDict_CheckExact(op)) {
            _PyDict_MaybeUntrack(op);
        }
        gc = next;
    }
}

/* Return true if object has a pre-PEP 442 finalization method. */
static int
has_legacy_finalizer(PyObject *op)
{
    return op->ob_type->tp_del != NULL;
}

/* Move the objects in unreachable with tp_del slots into `finalizers`.
 *
 * This function also removes NEXT_MASK_UNREACHABLE flag
 * from _gc_next in unreachable.
 */
static void
move_legacy_finalizers(PyGC_Head *unreachable, PyGC_Head *finalizers)
{
    PyGC_Head *gc, *next;
    unreachable->_gc_next &= ~NEXT_MASK_UNREACHABLE;

    /* March over unreachable.  Move objects with finalizers into
     * `finalizers`.
     */
    for (gc = GC_NEXT(unreachable); gc != unreachable; gc = next) {
        PyObject *op = FROM_GC(gc);

        _PyObject_ASSERT(op, gc->_gc_next & NEXT_MASK_UNREACHABLE);
        gc->_gc_next &= ~NEXT_MASK_UNREACHABLE;
        next = (PyGC_Head*)gc->_gc_next;

        if (has_legacy_finalizer(op)) {
            gc_clear_collecting(gc);
            gc_list_move(gc, finalizers);
        }
    }
}

/* A traversal callback for move_legacy_finalizer_reachable. */
static int
visit_move(PyObject *op, PyGC_Head *tolist)
{
    if (PyObject_IS_GC(op)) {
        PyGC_Head *gc = AS_GC(op);
        if (gc_is_collecting(gc)) {
            gc_list_move(gc, tolist);
            gc_clear_collecting(gc);
        }
    }
    return 0;
}

/* Move objects that are reachable from finalizers, from the unreachable set
 * into finalizers set.
 */
static void
move_legacy_finalizer_reachable(PyGC_Head *finalizers)
{
    traverseproc traverse;
    PyGC_Head *gc = GC_NEXT(finalizers);
    for (; gc != finalizers; gc = GC_NEXT(gc)) {
        /* Note that the finalizers list may grow during this. */
        traverse = Py_TYPE(FROM_GC(gc))->tp_traverse;
        (void) traverse(FROM_GC(gc),
                        (visitproc)visit_move,
                        (void *)finalizers);
    }
}

/* Clear all weakrefs to unreachable objects, and if such a weakref has a
 * callback, invoke it if necessary.  Note that it's possible for such
 * weakrefs to be outside the unreachable set -- indeed, those are precisely
 * the weakrefs whose callbacks must be invoked.  See gc_weakref.txt for
 * overview & some details.  Some weakrefs with callbacks may be reclaimed
 * directly by this routine; the number reclaimed is the return value.  Other
 * weakrefs with callbacks may be moved into the `old` generation.  Objects
 * moved into `old` have gc_refs set to GC_REACHABLE; the objects remaining in
 * unreachable are left at GC_TENTATIVELY_UNREACHABLE.  When this returns,
 * no object in `unreachable` is weakly referenced anymore.
 */
static int
handle_weakrefs(PyGC_Head *unreachable, PyGC_Head *old)
{
    PyGC_Head *gc;
    PyObject *op;               /* generally FROM_GC(gc) */
    PyWeakReference *wr;        /* generally a cast of op */
    PyGC_Head wrcb_to_call;     /* weakrefs with callbacks to call */
    PyGC_Head *next;
    int num_freed = 0;

    gc_list_init(&wrcb_to_call);

    /* Clear all weakrefs to the objects in unreachable.  If such a weakref
     * also has a callback, move it into `wrcb_to_call` if the callback
     * needs to be invoked.  Note that we cannot invoke any callbacks until
     * all weakrefs to unreachable objects are cleared, lest the callback
     * resurrect an unreachable object via a still-active weakref.  We
     * make another pass over wrcb_to_call, invoking callbacks, after this
     * pass completes.
     */
    for (gc = GC_NEXT(unreachable); gc != unreachable; gc = next) {
        PyWeakReference **wrlist;

        op = FROM_GC(gc);
        next = GC_NEXT(gc);

        if (PyWeakref_Check(op)) {
            /* A weakref inside the unreachable set must be cleared.  If we
             * allow its callback to execute inside delete_garbage(), it
             * could expose objects that have tp_clear already called on
             * them.  Or, it could resurrect unreachable objects.  One way
             * this can happen is if some container objects do not implement
             * tp_traverse.  Then, wr_object can be outside the unreachable
             * set but can be deallocated as a result of breaking the
             * reference cycle.  If we don't clear the weakref, the callback
             * will run and potentially cause a crash.  See bpo-38006 for
             * one example.
             */
            _PyWeakref_ClearRef((PyWeakReference *)op);
        }

        if (! PyType_SUPPORTS_WEAKREFS(Py_TYPE(op)))
            continue;

        /* It supports weakrefs.  Does it have any? */
        wrlist = (PyWeakReference **)
                                PyObject_GET_WEAKREFS_LISTPTR(op);

        /* `op` may have some weakrefs.  March over the list, clear
         * all the weakrefs, and move the weakrefs with callbacks
         * that must be called into wrcb_to_call.
         */
        for (wr = *wrlist; wr != NULL; wr = *wrlist) {
            PyGC_Head *wrasgc;                  /* AS_GC(wr) */

            /* _PyWeakref_ClearRef clears the weakref but leaves
             * the callback pointer intact.  Obscure:  it also
             * changes *wrlist.
             */
            _PyObject_ASSERT((PyObject *)wr, wr->wr_object == op);
            _PyWeakref_ClearRef(wr);
            _PyObject_ASSERT((PyObject *)wr, wr->wr_object == Py_None);
            if (wr->wr_callback == NULL) {
                /* no callback */
                continue;
            }

            /* Headache time.  `op` is going away, and is weakly referenced by
             * `wr`, which has a callback.  Should the callback be invoked?  If wr
             * is also trash, no:
             *
             * 1. There's no need to call it.  The object and the weakref are
             *    both going away, so it's legitimate to pretend the weakref is
             *    going away first.  The user has to ensure a weakref outlives its
             *    referent if they want a guarantee that the wr callback will get
             *    invoked.
             *
             * 2. It may be catastrophic to call it.  If the callback is also in
             *    cyclic trash (CT), then although the CT is unreachable from
             *    outside the current generation, CT may be reachable from the
             *    callback.  Then the callback could resurrect insane objects.
             *
             * Since the callback is never needed and may be unsafe in this case,
             * wr is simply left in the unreachable set.  Note that because we
             * already called _PyWeakref_ClearRef(wr), its callback will never
             * trigger.
             *
             * OTOH, if wr isn't part of CT, we should invoke the callback:  the
             * weakref outlived the trash.  Note that since wr isn't CT in this
             * case, its callback can't be CT either -- wr acted as an external
             * root to this generation, and therefore its callback did too.  So
             * nothing in CT is reachable from the callback either, so it's hard
             * to imagine how calling it later could create a problem for us.  wr
             * is moved to wrcb_to_call in this case.
             */
            if (gc_is_collecting(AS_GC(wr))) {
                /* it should already have been cleared above */
                assert(wr->wr_object == Py_None);
                continue;
            }

            /* Create a new reference so that wr can't go away
             * before we can process it again.
             */
            Py_INCREF(wr);

            /* Move wr to wrcb_to_call, for the next pass. */
            wrasgc = AS_GC(wr);
            assert(wrasgc != next); /* wrasgc is reachable, but
                                       next isn't, so they can't
                                       be the same */
            gc_list_move(wrasgc, &wrcb_to_call);
        }
    }

    /* Invoke the callbacks we decided to honor.  It's safe to invoke them
     * because they can't reference unreachable objects.
     */
    while (! gc_list_is_empty(&wrcb_to_call)) {
        PyObject *temp;
        PyObject *callback;

        gc = (PyGC_Head*)wrcb_to_call._gc_next;
        op = FROM_GC(gc);
        _PyObject_ASSERT(op, PyWeakref_Check(op));
        wr = (PyWeakReference *)op;
        callback = wr->wr_callback;
        _PyObject_ASSERT(op, callback != NULL);

        /* copy-paste of weakrefobject.c's handle_callback() */
        temp = PyObject_CallFunctionObjArgs(callback, wr, NULL);
        if (temp == NULL)
            PyErr_WriteUnraisable(callback);
        else
            Py_DECREF(temp);

        /* Give up the reference we created in the first pass.  When
         * op's refcount hits 0 (which it may or may not do right now),
         * op's tp_dealloc will decref op->wr_callback too.  Note
         * that the refcount probably will hit 0 now, and because this
         * weakref was reachable to begin with, gc didn't already
         * add it to its count of freed objects.  Example:  a reachable
         * weak value dict maps some key to this reachable weakref.
         * The callback removes this key->weakref mapping from the
         * dict, leaving no other references to the weakref (excepting
         * ours).
         */
        Py_DECREF(op);
        if (wrcb_to_call._gc_next == (uintptr_t)gc) {
            /* object is still alive -- move it */
            gc_list_move(gc, old);
        }
        else {
            ++num_freed;
        }
    }

    return num_freed;
}

static void
debug_cycle(const char *msg, PyObject *op)
{
    PySys_FormatStderr("gc: %s <%s %p>\n",
                       msg, Py_TYPE(op)->tp_name, op);
}

/* Handle uncollectable garbage (cycles with tp_del slots, and stuff reachable
 * only from such cycles).
 * If DEBUG_SAVEALL, all objects in finalizers are appended to the module
 * garbage list (a Python list), else only the objects in finalizers with
 * __del__ methods are appended to garbage.  All objects in finalizers are
 * merged into the old list regardless.
 */
static void
handle_legacy_finalizers(struct _gc_runtime_state *state,
                         PyGC_Head *finalizers, PyGC_Head *old)
{
    assert(!PyErr_Occurred());

    PyGC_Head *gc = GC_NEXT(finalizers);
    if (state->garbage == NULL) {
        state->garbage = PyList_New(0);
        if (state->garbage == NULL)
            Py_FatalError("gc couldn't create gc.garbage list");
    }
    for (; gc != finalizers; gc = GC_NEXT(gc)) {
        PyObject *op = FROM_GC(gc);

        if ((state->debug & DEBUG_SAVEALL) || has_legacy_finalizer(op)) {
            if (PyList_Append(state->garbage, op) < 0) {
                PyErr_Clear();
                break;
            }
        }
    }

    gc_list_merge(finalizers, old);
}

/* Run first-time finalizers (if any) on all the objects in collectable.
 * Note that this may remove some (or even all) of the objects from the
 * list, due to refcounts falling to 0.
 */
static void
finalize_garbage(PyGC_Head *collectable)
{
    destructor finalize;
    PyGC_Head seen;

    /* While we're going through the loop, `finalize(op)` may cause op, or
     * other objects, to be reclaimed via refcounts falling to zero.  So
     * there's little we can rely on about the structure of the input
     * `collectable` list across iterations.  For safety, we always take the
     * first object in that list and move it to a temporary `seen` list.
     * If objects vanish from the `collectable` and `seen` lists we don't
     * care.
     */
    gc_list_init(&seen);

    while (!gc_list_is_empty(collectable)) {
        PyGC_Head *gc = GC_NEXT(collectable);
        PyObject *op = FROM_GC(gc);
        gc_list_move(gc, &seen);
        if (!_PyGCHead_FINALIZED(gc) &&
                (finalize = Py_TYPE(op)->tp_finalize) != NULL) {
            _PyGCHead_SET_FINALIZED(gc);
            Py_INCREF(op);
            finalize(op);
            assert(!PyErr_Occurred());
            Py_DECREF(op);
        }
    }
    gc_list_merge(&seen, collectable);
}

/* Walk the collectable list and check that they are really unreachable
   from the outside (some objects could have been resurrected by a
   finalizer). */
static int
check_garbage(PyGC_Head *collectable)
{
    int ret = 0;
    PyGC_Head *gc;
    for (gc = GC_NEXT(collectable); gc != collectable; gc = GC_NEXT(gc)) {
        // Use gc_refs and break gc_prev again.
        gc_set_refs(gc, Py_REFCNT(FROM_GC(gc)));
        _PyObject_ASSERT(FROM_GC(gc), gc_get_refs(gc) != 0);
    }
    subtract_refs(collectable);
    PyGC_Head *prev = collectable;
    for (gc = GC_NEXT(collectable); gc != collectable; gc = GC_NEXT(gc)) {
        _PyObject_ASSERT_WITH_MSG(FROM_GC(gc),
                                  gc_get_refs(gc) >= 0,
                                  "refcount is too small");
        if (gc_get_refs(gc) != 0) {
            ret = -1;
        }
        // Restore gc_prev here.
        _PyGCHead_SET_PREV(gc, prev);
        gc_clear_collecting(gc);
        prev = gc;
    }
    return ret;
}

/* Break reference cycles by clearing the containers involved.  This is
 * tricky business as the lists can be changing and we don't know which
 * objects may be freed.  It is possible I screwed something up here.
 */
static void
delete_garbage(struct _gc_runtime_state *state,
               PyGC_Head *collectable, PyGC_Head *old)
{
    assert(!PyErr_Occurred());

    while (!gc_list_is_empty(collectable)) {
        PyGC_Head *gc = GC_NEXT(collectable);
        PyObject *op = FROM_GC(gc);

        _PyObject_ASSERT_WITH_MSG(op, Py_REFCNT(op) > 0,
                                  "refcount is too small");

        if (state->debug & DEBUG_SAVEALL) {
            assert(state->garbage != NULL);
            if (PyList_Append(state->garbage, op) < 0) {
                PyErr_Clear();
            }
        }
        else {
            inquiry clear;
            if ((clear = Py_TYPE(op)->tp_clear) != NULL) {
                Py_INCREF(op);
                (void) clear(op);
                if (PyErr_Occurred()) {
                    _PyErr_WriteUnraisableMsg("in tp_clear of",
                                              (PyObject*)Py_TYPE(op));
                }
                Py_DECREF(op);
            }
        }
        if (GC_NEXT(collectable) == gc) {
            /* object is still alive, move it, it may die later */
            gc_list_move(gc, old);
        }
    }
}

/* Clear all free lists
 * All free lists are cleared during the collection of the highest generation.
 * Allocated items in the free list may keep a pymalloc arena occupied.
 * Clearing the free lists may give back memory to the OS earlier.
 */
static void
clear_freelists(void)
{
    (void)PyMethod_ClearFreeList();
    (void)PyFrame_ClearFreeList();
    (void)PyCFunction_ClearFreeList();
    (void)PyTuple_ClearFreeList();
    (void)PyUnicode_ClearFreeList();
    (void)PyFloat_ClearFreeList();
    (void)PyList_ClearFreeList();
    (void)PyDict_ClearFreeList();
    (void)PySet_ClearFreeList();
    (void)PyAsyncGen_ClearFreeLists();
    (void)PyContext_ClearFreeList();
}

// Show stats for objects in each gennerations.
static void
show_stats_each_generations(struct _gc_runtime_state *state)
{
    char buf[100];
    size_t pos = 0;

    for (int i = 0; i < NUM_GENERATIONS && pos < sizeof(buf); i++) {
        pos += PyOS_snprintf(buf+pos, sizeof(buf)-pos,
                             " %"PY_FORMAT_SIZE_T"d",
                             gc_list_size(GEN_HEAD(state, i)));
    }

    PySys_FormatStderr(
        "gc: objects in each generation:%s\n"
        "gc: objects in permanent generation: %zd\n",
        buf, gc_list_size(&state->permanent_generation.head));
}

/* This is the main function.  Read this to understand how the
 * collection process works. */
static Py_ssize_t
collect(struct _gc_runtime_state *state, int generation,
        Py_ssize_t *n_collected, Py_ssize_t *n_uncollectable, int nofail)
{
    int i;
    Py_ssize_t m = 0; /* # objects collected */
    Py_ssize_t n = 0; /* # unreachable objects that couldn't be collected */
    PyGC_Head *young; /* the generation we are examining */
    PyGC_Head *old; /* next older generation */
    PyGC_Head unreachable; /* non-problematic unreachable trash */
    PyGC_Head finalizers;  /* objects with, & reachable from, __del__ */
    PyGC_Head *gc;
    _PyTime_t t1 = 0;   /* initialize to prevent a compiler warning */

    if (state->debug & DEBUG_STATS) {
        PySys_WriteStderr("gc: collecting generation %d...\n", generation);
        show_stats_each_generations(state);
        t1 = _PyTime_GetMonotonicClock();
    }

    if (PyDTrace_GC_START_ENABLED())
        PyDTrace_GC_START(generation);

    /* update collection and allocation counters */
    if (generation+1 < NUM_GENERATIONS)
        state->generations[generation+1].count += 1;
    for (i = 0; i <= generation; i++)
        state->generations[i].count = 0;

    /* merge younger generations with one we are currently collecting */
    for (i = 0; i < generation; i++) {
        gc_list_merge(GEN_HEAD(state, i), GEN_HEAD(state, generation));
    }

    /* handy references */
    young = GEN_HEAD(state, generation);
    if (generation < NUM_GENERATIONS-1)
        old = GEN_HEAD(state, generation+1);
    else
        old = young;

    validate_list(young, 0);
    validate_list(old, 0);
    /* Using ob_refcnt and gc_refs, calculate which objects in the
     * container set are reachable from outside the set (i.e., have a
     * refcount greater than 0 when all the references within the
     * set are taken into account).
     */
    update_refs(young);  // gc_prev is used for gc_refs
    subtract_refs(young);

    /* Leave everything reachable from outside young in young, and move
     * everything else (in young) to unreachable.
     * NOTE:  This used to move the reachable objects into a reachable
     * set instead.  But most things usually turn out to be reachable,
     * so it's more efficient to move the unreachable things.
     */
    gc_list_init(&unreachable);
    move_unreachable(young, &unreachable);  // gc_prev is pointer again
    validate_list(young, 0);

    untrack_tuples(young);
    /* Move reachable objects to next generation. */
    if (young != old) {
        if (generation == NUM_GENERATIONS - 2) {
            state->long_lived_pending += gc_list_size(young);
        }
        gc_list_merge(young, old);
    }
    else {
        /* We only untrack dicts in full collections, to avoid quadratic
           dict build-up. See issue #14775. */
        untrack_dicts(young);
        state->long_lived_pending = 0;
        state->long_lived_total = gc_list_size(young);
    }

    /* All objects in unreachable are trash, but objects reachable from
     * legacy finalizers (e.g. tp_del) can't safely be deleted.
     */
    gc_list_init(&finalizers);
    // NEXT_MASK_UNREACHABLE is cleared here.
    // After move_legacy_finalizers(), unreachable is normal list.
    move_legacy_finalizers(&unreachable, &finalizers);
    /* finalizers contains the unreachable objects with a legacy finalizer;
     * unreachable objects reachable *from* those are also uncollectable,
     * and we move those into the finalizers list too.
     */
    move_legacy_finalizer_reachable(&finalizers);

    validate_list(&finalizers, 0);
    validate_list(&unreachable, PREV_MASK_COLLECTING);

    /* Print debugging information. */
    if (state->debug & DEBUG_COLLECTABLE) {
        for (gc = GC_NEXT(&unreachable); gc != &unreachable; gc = GC_NEXT(gc)) {
            debug_cycle("collectable", FROM_GC(gc));
        }
    }

    /* Clear weakrefs and invoke callbacks as necessary. */
    m += handle_weakrefs(&unreachable, old);

    validate_list(old, 0);
    validate_list(&unreachable, PREV_MASK_COLLECTING);

    /* Call tp_finalize on objects which have one. */
    finalize_garbage(&unreachable);

    if (check_garbage(&unreachable)) { // clear PREV_MASK_COLLECTING here
        gc_list_merge(&unreachable, old);
    }
    else {
        /* Call tp_clear on objects in the unreachable set.  This will cause
         * the reference cycles to be broken.  It may also cause some objects
         * in finalizers to be freed.
         */
        m += gc_list_size(&unreachable);
        delete_garbage(state, &unreachable, old);
    }

    /* Collect statistics on uncollectable objects found and print
     * debugging information. */
    for (gc = GC_NEXT(&finalizers); gc != &finalizers; gc = GC_NEXT(gc)) {
        n++;
        if (state->debug & DEBUG_UNCOLLECTABLE)
            debug_cycle("uncollectable", FROM_GC(gc));
    }
    if (state->debug & DEBUG_STATS) {
        double d = _PyTime_AsSecondsDouble(_PyTime_GetMonotonicClock() - t1);
        PySys_WriteStderr(
            "gc: done, %" PY_FORMAT_SIZE_T "d unreachable, "
            "%" PY_FORMAT_SIZE_T "d uncollectable, %.4fs elapsed\n",
            n+m, n, d);
    }

    /* Append instances in the uncollectable set to a Python
     * reachable list of garbage.  The programmer has to deal with
     * this if they insist on creating this type of structure.
     */
    handle_legacy_finalizers(state, &finalizers, old);
    validate_list(old, 0);

    /* Clear free list only during the collection of the highest
     * generation */
    if (generation == NUM_GENERATIONS-1) {
        clear_freelists();
    }

    if (PyErr_Occurred()) {
        if (nofail) {
            PyErr_Clear();
        }
        else {
            if (gc_str == NULL)
                gc_str = PyUnicode_FromString("garbage collection");
            PyErr_WriteUnraisable(gc_str);
            Py_FatalError("unexpected exception during garbage collection");
        }
    }

    /* Update stats */
    if (n_collected) {
        *n_collected = m;
    }
    if (n_uncollectable) {
        *n_uncollectable = n;
    }

    struct gc_generation_stats *stats = &state->generation_stats[generation];
    stats->collections++;
    stats->collected += m;
    stats->uncollectable += n;

    if (PyDTrace_GC_DONE_ENABLED()) {
        PyDTrace_GC_DONE(n+m);
    }

    assert(!PyErr_Occurred());
    return n+m;
}

/* Invoke progress callbacks to notify clients that garbage collection
 * is starting or stopping
 */
static void
invoke_gc_callback(struct _gc_runtime_state *state, const char *phase,
                   int generation, Py_ssize_t collected,
                   Py_ssize_t uncollectable)
{
    assert(!PyErr_Occurred());

    /* we may get called very early */
    if (state->callbacks == NULL) {
        return;
    }

    /* The local variable cannot be rebound, check it for sanity */
    assert(PyList_CheckExact(state->callbacks));
    PyObject *info = NULL;
    if (PyList_GET_SIZE(state->callbacks) != 0) {
        info = Py_BuildValue("{sisnsn}",
            "generation", generation,
            "collected", collected,
            "uncollectable", uncollectable);
        if (info == NULL) {
            PyErr_WriteUnraisable(NULL);
            return;
        }
    }
    for (Py_ssize_t i=0; i<PyList_GET_SIZE(state->callbacks); i++) {
        PyObject *r, *cb = PyList_GET_ITEM(state->callbacks, i);
        Py_INCREF(cb); /* make sure cb doesn't go away */
        r = PyObject_CallFunction(cb, "sO", phase, info);
        if (r == NULL) {
            PyErr_WriteUnraisable(cb);
        }
        else {
            Py_DECREF(r);
        }
        Py_DECREF(cb);
    }
    Py_XDECREF(info);
    assert(!PyErr_Occurred());
}

/* Perform garbage collection of a generation and invoke
 * progress callbacks.
 */
static Py_ssize_t
collect_with_callback(struct _gc_runtime_state *state, int generation)
{
    assert(!PyErr_Occurred());
    Py_ssize_t result, collected, uncollectable;
    invoke_gc_callback(state, "start", generation, 0, 0);
    result = collect(state, generation, &collected, &uncollectable, 0);
    invoke_gc_callback(state, "stop", generation, collected, uncollectable);
    assert(!PyErr_Occurred());
    return result;
}

//static Py_ssize_t
//collect_generations(struct _gc_runtime_state *state)
//{
//    /* Find the oldest generation (highest numbered) where the count
//     * exceeds the threshold.  Objects in the that generation and
//     * generations younger than it will be collected. */
//    Py_ssize_t n = 0;
//    for (int i = NUM_GENERATIONS-1; i >= 0; i--) {
//        if (state->generations[i].count > state->generations[i].threshold) {
//            /* Avoid quadratic performance degradation in number
//               of tracked objects. See comments at the beginning
//               of this file, and issue #4074.
//            */
//            if (i == NUM_GENERATIONS - 1
//                && state->long_lived_pending < state->long_lived_total / 4)
//                continue;
//            n = collect_with_callback(state, i);
//            break;
//        }
//    }
//    return n;
//}

#include "clinic/gcmodule.c.h"

/*[clinic input]
gc.enable

Enable automatic garbage collection.
[clinic start generated code]*/

static PyObject *
gc_enable_impl(PyObject *module)
/*[clinic end generated code: output=45a427e9dce9155c input=81ac4940ca579707]*/
{
    _PyRuntime.gc.enabled = 1;
    Py_RETURN_NONE;
}

/*[clinic input]
gc.disable

Disable automatic garbage collection.
[clinic start generated code]*/

static PyObject *
gc_disable_impl(PyObject *module)
/*[clinic end generated code: output=97d1030f7aa9d279 input=8c2e5a14e800d83b]*/
{
    _PyRuntime.gc.enabled = 0;
    Py_RETURN_NONE;
}

/*[clinic input]
gc.isenabled -> bool

Returns true if automatic garbage collection is enabled.
[clinic start generated code]*/

static int
gc_isenabled_impl(PyObject *module)
/*[clinic end generated code: output=1874298331c49130 input=30005e0422373b31]*/
{
    return _PyRuntime.gc.enabled;
}

/*[clinic input]
gc.collect -> Py_ssize_t

    generation: int(c_default="NUM_GENERATIONS - 1") = 2

Run the garbage collector.

With no arguments, run a full collection.  The optional argument
may be an integer specifying which generation to collect.  A ValueError
is raised if the generation number is invalid.

The number of unreachable objects is returned.
[clinic start generated code]*/

static Py_ssize_t
gc_collect_impl(PyObject *module, int generation)
/*[clinic end generated code: output=b697e633043233c7 input=40720128b682d879]*/
{

    if (generation < 0 || generation >= NUM_GENERATIONS) {
        PyErr_SetString(PyExc_ValueError, "invalid generation");
        return -1;
    }

    struct _gc_runtime_state *state = &_PyRuntime.gc;
    Py_ssize_t n;
    if (state->collecting) {
        /* already collecting, don't do anything */
        n = 0;
    }
    else {
        state->collecting = 1;
        n = collect_with_callback(state, generation);
        state->collecting = 0;
    }
    return n;
}

/*[clinic input]
gc.set_debug

    flags: int
        An integer that can have the following bits turned on:
          DEBUG_STATS - Print statistics during collection.
          DEBUG_COLLECTABLE - Print collectable objects found.
          DEBUG_UNCOLLECTABLE - Print unreachable but uncollectable objects
            found.
          DEBUG_SAVEALL - Save objects to gc.garbage rather than freeing them.
          DEBUG_LEAK - Debug leaking programs (everything but STATS).
    /

Set the garbage collection debugging flags.

Debugging information is written to sys.stderr.
[clinic start generated code]*/

static PyObject *
gc_set_debug_impl(PyObject *module, int flags)
/*[clinic end generated code: output=7c8366575486b228 input=5e5ce15e84fbed15]*/
{
    _PyRuntime.gc.debug = flags;

    Py_RETURN_NONE;
}

/*[clinic input]
gc.get_debug -> int

Get the garbage collection debugging flags.
[clinic start generated code]*/

static int
gc_get_debug_impl(PyObject *module)
/*[clinic end generated code: output=91242f3506cd1e50 input=91a101e1c3b98366]*/
{
    return _PyRuntime.gc.debug;
}

PyDoc_STRVAR(gc_set_thresh__doc__,
"set_threshold(threshold0, [threshold1, threshold2]) -> None\n"
"\n"
"Sets the collection thresholds.  Setting threshold0 to zero disables\n"
"collection.\n");

static PyObject *
gc_set_threshold(PyObject *self, PyObject *args)
{
    struct _gc_runtime_state *state = &_PyRuntime.gc;
    if (!PyArg_ParseTuple(args, "i|ii:set_threshold",
                          &state->generations[0].threshold,
                          &state->generations[1].threshold,
                          &state->generations[2].threshold))
        return NULL;
    for (int i = 3; i < NUM_GENERATIONS; i++) {
        /* generations higher than 2 get the same threshold */
        state->generations[i].threshold = state->generations[2].threshold;
    }
    Py_RETURN_NONE;
}

/*[clinic input]
gc.get_threshold

Return the current collection thresholds.
[clinic start generated code]*/

static PyObject *
gc_get_threshold_impl(PyObject *module)
/*[clinic end generated code: output=7902bc9f41ecbbd8 input=286d79918034d6e6]*/
{
    struct _gc_runtime_state *state = &_PyRuntime.gc;
    return Py_BuildValue("(iii)",
                         state->generations[0].threshold,
                         state->generations[1].threshold,
                         state->generations[2].threshold);
}

/*[clinic input]
gc.get_count

Return a three-tuple of the current collection counts.
[clinic start generated code]*/

static PyObject *
gc_get_count_impl(PyObject *module)
/*[clinic end generated code: output=354012e67b16398f input=a392794a08251751]*/
{
    struct _gc_runtime_state *state = &_PyRuntime.gc;
    return Py_BuildValue("(iii)",
                         state->generations[0].count,
                         state->generations[1].count,
                         state->generations[2].count);
}

static int
referrersvisit(PyObject* obj, PyObject *objs)
{
    Py_ssize_t i;
    for (i = 0; i < PyTuple_GET_SIZE(objs); i++)
        if (PyTuple_GET_ITEM(objs, i) == obj)
            return 1;
    return 0;
}

static int
gc_referrers_for(PyObject *objs, PyGC_Head *list, PyObject *resultlist)
{
    PyGC_Head *gc;
    PyObject *obj;
    traverseproc traverse;
    for (gc = GC_NEXT(list); gc != list; gc = GC_NEXT(gc)) {
        obj = FROM_GC(gc);
        traverse = Py_TYPE(obj)->tp_traverse;
        if (obj == objs || obj == resultlist)
            continue;
        if (traverse(obj, (visitproc)referrersvisit, objs)) {
            if (PyList_Append(resultlist, obj) < 0)
                return 0; /* error */
        }
    }
    return 1; /* no error */
}

PyDoc_STRVAR(gc_get_referrers__doc__,
"get_referrers(*objs) -> list\n\
Return the list of objects that directly refer to any of objs.");

static PyObject *
gc_get_referrers(PyObject *self, PyObject *args)
{
    int i;
    PyObject *result = PyList_New(0);
    if (!result) return NULL;

    struct _gc_runtime_state *state = &_PyRuntime.gc;
    for (i = 0; i < NUM_GENERATIONS; i++) {
        if (!(gc_referrers_for(args, GEN_HEAD(state, i), result))) {
            Py_DECREF(result);
            return NULL;
        }
    }
    return result;
}

/* Append obj to list; return true if error (out of memory), false if OK. */
static int
referentsvisit(PyObject *obj, PyObject *list)
{
    return PyList_Append(list, obj) < 0;
}

PyDoc_STRVAR(gc_get_referents__doc__,
"get_referents(*objs) -> list\n\
Return the list of objects that are directly referred to by objs.");

static PyObject *
gc_get_referents(PyObject *self, PyObject *args)
{
    Py_ssize_t i;
    PyObject *result = PyList_New(0);

    if (result == NULL)
        return NULL;

    for (i = 0; i < PyTuple_GET_SIZE(args); i++) {
        traverseproc traverse;
        PyObject *obj = PyTuple_GET_ITEM(args, i);

        if (! PyObject_IS_GC(obj))
            continue;
        traverse = Py_TYPE(obj)->tp_traverse;
        if (! traverse)
            continue;
        if (traverse(obj, (visitproc)referentsvisit, result)) {
            Py_DECREF(result);
            return NULL;
        }
    }
    return result;
}

/*[clinic input]
gc.get_objects
    generation: Py_ssize_t(accept={int, NoneType}, c_default="-1") = None
        Generation to extract the objects from.

Return a list of objects tracked by the collector (excluding the list returned).

If generation is not None, return only the objects tracked by the collector
that are in that generation.
[clinic start generated code]*/

static PyObject *
gc_get_objects_impl(PyObject *module, Py_ssize_t generation)
/*[clinic end generated code: output=48b35fea4ba6cb0e input=ef7da9df9806754c]*/
{
    int i;
    PyObject* result;
    struct _gc_runtime_state *state = &_PyRuntime.gc;

    result = PyList_New(0);
    if (result == NULL) {
        return NULL;
    }

    /* If generation is passed, we extract only that generation */
    if (generation != -1) {
        if (generation >= NUM_GENERATIONS) {
            PyErr_Format(PyExc_ValueError,
                         "generation parameter must be less than the number of "
                         "available generations (%i)",
                          NUM_GENERATIONS);
            goto error;
        }

        if (generation < 0) {
            PyErr_SetString(PyExc_ValueError,
                            "generation parameter cannot be negative");
            goto error;
        }

        if (append_objects(result, GEN_HEAD(state, generation))) {
            goto error;
        }

        return result;
    }

    /* If generation is not passed or None, get all objects from all generations */
    for (i = 0; i < NUM_GENERATIONS; i++) {
        if (append_objects(result, GEN_HEAD(state, i))) {
            goto error;
        }
    }
    return result;

error:
    Py_DECREF(result);
    return NULL;
}

/*[clinic input]
gc.get_stats

Return a list of dictionaries containing per-generation statistics.
[clinic start generated code]*/

static PyObject *
gc_get_stats_impl(PyObject *module)
/*[clinic end generated code: output=a8ab1d8a5d26f3ab input=1ef4ed9d17b1a470]*/
{
    int i;
    struct gc_generation_stats stats[NUM_GENERATIONS], *st;

    /* To get consistent values despite allocations while constructing
       the result list, we use a snapshot of the running stats. */
    struct _gc_runtime_state *state = &_PyRuntime.gc;
    for (i = 0; i < NUM_GENERATIONS; i++) {
        stats[i] = state->generation_stats[i];
    }

    PyObject *result = PyList_New(0);
    if (result == NULL)
        return NULL;

    for (i = 0; i < NUM_GENERATIONS; i++) {
        PyObject *dict;
        st = &stats[i];
        dict = Py_BuildValue("{snsnsn}",
                             "collections", st->collections,
                             "collected", st->collected,
                             "uncollectable", st->uncollectable
                            );
        if (dict == NULL)
            goto error;
        if (PyList_Append(result, dict)) {
            Py_DECREF(dict);
            goto error;
        }
        Py_DECREF(dict);
    }
    return result;

error:
    Py_XDECREF(result);
    return NULL;
}


/*[clinic input]
gc.is_tracked

    obj: object
    /

Returns true if the object is tracked by the garbage collector.

Simple atomic objects will return false.
[clinic start generated code]*/

static PyObject *
gc_is_tracked(PyObject *module, PyObject *obj)
/*[clinic end generated code: output=14f0103423b28e31 input=d83057f170ea2723]*/
{
    PyObject *result;

    if (PyObject_IS_GC(obj) && _PyObject_GC_IS_TRACKED(obj))
        result = Py_True;
    else
        result = Py_False;
    Py_INCREF(result);
    return result;
}

/*[clinic input]
gc.freeze

Freeze all current tracked objects and ignore them for future collections.

This can be used before a POSIX fork() call to make the gc copy-on-write friendly.
Note: collection before a POSIX fork() call may free pages for future allocation
which can cause copy-on-write.
[clinic start generated code]*/

static PyObject *
gc_freeze_impl(PyObject *module)
/*[clinic end generated code: output=502159d9cdc4c139 input=b602b16ac5febbe5]*/
{
    struct _gc_runtime_state *state = &_PyRuntime.gc;
    for (int i = 0; i < NUM_GENERATIONS; ++i) {
        gc_list_merge(GEN_HEAD(state, i), &state->permanent_generation.head);
        state->generations[i].count = 0;
    }
    Py_RETURN_NONE;
}

/*[clinic input]
gc.unfreeze

Unfreeze all objects in the permanent generation.

Put all objects in the permanent generation back into oldest generation.
[clinic start generated code]*/

static PyObject *
gc_unfreeze_impl(PyObject *module)
/*[clinic end generated code: output=1c15f2043b25e169 input=2dd52b170f4cef6c]*/
{
    struct _gc_runtime_state *state = &_PyRuntime.gc;
    gc_list_merge(&state->permanent_generation.head, GEN_HEAD(state, NUM_GENERATIONS-1));
    Py_RETURN_NONE;
}

/*[clinic input]
gc.get_freeze_count -> Py_ssize_t

Return the number of objects in the permanent generation.
[clinic start generated code]*/

static Py_ssize_t
gc_get_freeze_count_impl(PyObject *module)
/*[clinic end generated code: output=61cbd9f43aa032e1 input=45ffbc65cfe2a6ed]*/
{
    return gc_list_size(&_PyRuntime.gc.permanent_generation.head);
}

/*[clinic input]
gc.reward
    value: double
    /
Set the reward for the Q garbage collector.
[clinic start generated code]*/

static PyObject *
gc_reward_impl(PyObject *module, double value)
/*[clinic end generated code: output=efd2b9b189133062 input=155fd6d53e69e8b2]*/
{
    // Not yet enabled. Enable and return.
    if (!q_state.enabled) {
        // Print Q-learning parameters.
        printf("Q-Learning Parameters\n");
        printf("Q_REPLAY_SIZE: %lu\n", q_config.replay_size);
        printf("Q_LEARNING_RATE: %f\n", q_config.learning_rate);
        printf("Q_EPSILON_START: %f\n", q_config.epsilon_start);
        printf("Q_EPSILON_END: %f\n", q_config.epsilon_end);
        printf("Q_EPSILON_DECAY: %f\n", q_config.epsilon_decay);
        printf("Q_GAMMA: %f\n", q_config.gamma);
        printf("Q_SKIP: %lu\n", q_config.skip);
        printf("Q_MAX_MEMORY: %lu\n", q_config.max_memory);
        printf("Q_FENCE_MEMORY: %lu\n", q_config.fence_memory);
        printf("Q_FENCE_PENALTY: %f\n", q_config.fence_penalty);
        printf("Q_MEMORY_STEP: %u\n", q_config.memory_step);

        // Enable.
        q_state.enabled = true;
        Py_RETURN_NONE;
    }

    // No new observations collected. Do nothing.
    if (q_state.last_replay_index == q_state.replay_index) {
        Py_RETURN_NONE;
    }

    // We did an observation pass.
    if (q_state.last_replay_index == 0) {
        printf("Saw %lu sites.\n", (q_state.replay_index - q_state.last_replay_index));
    }

    // Create thread if training is not already ongoing. We do not need to join here since the flag toggle is the last
    // thing that happens in the thread.
    if (!q_state.replay_locked) {
        // Lock the replay table.
        q_state.replay_locked = true;

        // Pass in argument.
        q_state.arg_reward = value;

        // Start training thread.
        pthread_create(&q_state.tid, NULL, q_train, NULL);
    }

    // Return none.
    Py_RETURN_NONE;
}

/*[clinic input]
gc.memory_usage -> Py_ssize_t
Return the current memory usage in bytes.
[clinic start generated code]*/

static Py_ssize_t
gc_memory_usage_impl(PyObject *module)
/*[clinic end generated code: output=6bf0a65d36800cc7 input=9dcbf3d29dfa5bd9]*/
{
    return memory_usage;
}

/*[clinic input]
gc.random_actions -> Py_ssize_t
Return the number of random actions taken.
[clinic start generated code]*/

static Py_ssize_t
gc_random_actions_impl(PyObject *module)
/*[clinic end generated code: output=133936503d7dbfa1 input=15be10bb87d5f818]*/
{
    return q_state.random_actions;
}


PyDoc_STRVAR(gc__doc__,
"This module provides access to the garbage collector for reference cycles.\n"
"\n"
"enable() -- Enable automatic garbage collection.\n"
"disable() -- Disable automatic garbage collection.\n"
"isenabled() -- Returns true if automatic collection is enabled.\n"
"collect() -- Do a full collection right now.\n"
"get_count() -- Return the current collection counts.\n"
"get_stats() -- Return list of dictionaries containing per-generation stats.\n"
"set_debug() -- Set debugging flags.\n"
"get_debug() -- Get debugging flags.\n"
"set_threshold() -- Set the collection thresholds.\n"
"get_threshold() -- Return the current the collection thresholds.\n"
"get_objects() -- Return a list of all objects tracked by the collector.\n"
"is_tracked() -- Returns true if a given object is tracked.\n"
"get_referrers() -- Return the list of objects that refer to an object.\n"
"get_referents() -- Return the list of objects that an object refers to.\n"
"freeze() -- Freeze all tracked objects and ignore them for future collections.\n"
"unfreeze() -- Unfreeze all objects in the permanent generation.\n"
"get_freeze_count() -- Return the number of objects in the permanent generation.\n");

static PyMethodDef GcMethods[] = {
    GC_ENABLE_METHODDEF
    GC_DISABLE_METHODDEF
    GC_ISENABLED_METHODDEF
    GC_SET_DEBUG_METHODDEF
    GC_GET_DEBUG_METHODDEF
    GC_GET_COUNT_METHODDEF
    {"set_threshold",  gc_set_threshold, METH_VARARGS, gc_set_thresh__doc__},
    GC_GET_THRESHOLD_METHODDEF
    GC_COLLECT_METHODDEF
    GC_GET_OBJECTS_METHODDEF
    GC_GET_STATS_METHODDEF
    GC_IS_TRACKED_METHODDEF
    {"get_referrers",  gc_get_referrers, METH_VARARGS,
        gc_get_referrers__doc__},
    {"get_referents",  gc_get_referents, METH_VARARGS,
        gc_get_referents__doc__},
    GC_FREEZE_METHODDEF
    GC_UNFREEZE_METHODDEF
    GC_GET_FREEZE_COUNT_METHODDEF
    GC_REWARD_METHODDEF
    GC_MEMORY_USAGE_METHODDEF
    GC_RANDOM_ACTIONS_METHODDEF
    {NULL,      NULL}           /* Sentinel */
};

static struct PyModuleDef gcmodule = {
    PyModuleDef_HEAD_INIT,
    "gc",              /* m_name */
    gc__doc__,         /* m_doc */
    -1,                /* m_size */
    GcMethods,         /* m_methods */
    NULL,              /* m_reload */
    NULL,              /* m_traverse */
    NULL,              /* m_clear */
    NULL               /* m_free */
};

PyMODINIT_FUNC
PyInit_gc(void)
{
    PyObject *m;

    m = PyModule_Create(&gcmodule);

    if (m == NULL) {
        return NULL;
    }

    struct _gc_runtime_state *state = &_PyRuntime.gc;
    if (state->garbage == NULL) {
        state->garbage = PyList_New(0);
        if (state->garbage == NULL)
            return NULL;
    }
    Py_INCREF(state->garbage);
    if (PyModule_AddObject(m, "garbage", state->garbage) < 0)
        return NULL;

    if (state->callbacks == NULL) {
        state->callbacks = PyList_New(0);
        if (state->callbacks == NULL)
            return NULL;
    }
    Py_INCREF(state->callbacks);
    if (PyModule_AddObject(m, "callbacks", state->callbacks) < 0)
        return NULL;

#define ADD_INT(NAME) if (PyModule_AddIntConstant(m, #NAME, NAME) < 0) return NULL
    ADD_INT(DEBUG_STATS);
    ADD_INT(DEBUG_COLLECTABLE);
    ADD_INT(DEBUG_UNCOLLECTABLE);
    ADD_INT(DEBUG_SAVEALL);
    ADD_INT(DEBUG_LEAK);
#undef ADD_INT
    return m;
}

/* API to invoke gc.collect() from C */
Py_ssize_t
PyGC_Collect(void)
{
    struct _gc_runtime_state *state = &_PyRuntime.gc;
    if (!state->enabled) {
        return 0;
    }

    Py_ssize_t n;
    if (state->collecting) {
        /* already collecting, don't do anything */
        n = 0;
    }
    else {
        PyObject *exc, *value, *tb;
        state->collecting = 1;
        PyErr_Fetch(&exc, &value, &tb);
        n = collect_with_callback(state, NUM_GENERATIONS - 1);
        PyErr_Restore(exc, value, tb);
        state->collecting = 0;
    }

    return n;
}

Py_ssize_t
_PyGC_CollectIfEnabled(void)
{
    return PyGC_Collect();
}

Py_ssize_t
_PyGC_CollectNoFail(void)
{
    assert(!PyErr_Occurred());

    struct _gc_runtime_state *state = &_PyRuntime.gc;
    Py_ssize_t n;

    /* Ideally, this function is only called on interpreter shutdown,
       and therefore not recursively.  Unfortunately, when there are daemon
       threads, a daemon thread can start a cyclic garbage collection
       during interpreter shutdown (and then never finish it).
       See http://bugs.python.org/issue8713#msg195178 for an example.
       */
    if (state->collecting) {
        n = 0;
    }
    else {
        state->collecting = 1;
        n = collect(state, NUM_GENERATIONS - 1, NULL, NULL, 1);
        state->collecting = 0;
    }
    return n;
}

void
_PyGC_DumpShutdownStats(_PyRuntimeState *runtime)
{
    struct _gc_runtime_state *state = &runtime->gc;
    if (!(state->debug & DEBUG_SAVEALL)
        && state->garbage != NULL && PyList_GET_SIZE(state->garbage) > 0) {
        const char *message;
        if (state->debug & DEBUG_UNCOLLECTABLE)
            message = "gc: %zd uncollectable objects at " \
                "shutdown";
        else
            message = "gc: %zd uncollectable objects at " \
                "shutdown; use gc.set_debug(gc.DEBUG_UNCOLLECTABLE) to list them";
        /* PyErr_WarnFormat does too many things and we are at shutdown,
           the warnings module's dependencies (e.g. linecache) may be gone
           already. */
        if (PyErr_WarnExplicitFormat(PyExc_ResourceWarning, "gc", 0,
                                     "gc", NULL, message,
                                     PyList_GET_SIZE(state->garbage)))
            PyErr_WriteUnraisable(NULL);
        if (state->debug & DEBUG_UNCOLLECTABLE) {
            PyObject *repr = NULL, *bytes = NULL;
            repr = PyObject_Repr(state->garbage);
            if (!repr || !(bytes = PyUnicode_EncodeFSDefault(repr)))
                PyErr_WriteUnraisable(state->garbage);
            else {
                PySys_WriteStderr(
                    "      %s\n",
                    PyBytes_AS_STRING(bytes)
                    );
            }
            Py_XDECREF(repr);
            Py_XDECREF(bytes);
        }
    }
}

void
_PyGC_Fini(_PyRuntimeState *runtime)
{
    struct _gc_runtime_state *state = &runtime->gc;
    Py_CLEAR(state->garbage);
    Py_CLEAR(state->callbacks);

    // Join training thread by release GIL, waiting, then reacquiring.
    PyGILState_Release(PyGILState_Ensure());
    pthread_join(q_state.tid, NULL);
    PyGILState_Ensure();

    // Print metadata.
    HTIterate(q_state.q_table, q_print_hashtable_entry, NULL);

    // Print aggregate table statistics.
    if (HTObjcnt(q_state.q_table) > 0) {
        printf("Total entries: %lu\n", HTObjcnt(q_state.q_table));
    }

    // Cleanup table.
    HTIterate(q_state.q_table, q_free_hashtable_entry, NULL);

    // Cleanup Q state.
    HTDestroy(q_state.q_table);
    PyMem_RawFree(q_state.replay);
}

/* for debugging */
void
_PyGC_Dump(PyGC_Head *g)
{
    _PyObject_Dump(FROM_GC(g));
}

/* extension modules might be compiled with GC support so these
   functions must always be available */

void
PyObject_GC_Track(void *op_raw)
{
    PyObject *op = _PyObject_CAST(op_raw);
    if (_PyObject_GC_IS_TRACKED(op)) {
        _PyObject_ASSERT_FAILED_MSG(op,
                                    "object already tracked "
                                    "by the garbage collector");
    }
    _PyObject_GC_TRACK(op);
}

void
PyObject_GC_UnTrack(void *op_raw)
{
    PyObject *op = _PyObject_CAST(op_raw);
    /* Obscure:  the Py_TRASHCAN mechanism requires that we be able to
     * call PyObject_GC_UnTrack twice on an object.
     */
    if (_PyObject_GC_IS_TRACKED(op)) {
        _PyObject_GC_UNTRACK(op);
    }
}

static PyObject *
_PyObject_GC_Alloc(int use_calloc, size_t basicsize)
{
    struct _gc_runtime_state *state = &_PyRuntime.gc;
    PyObject *op;
    PyGC_Head *g;
    size_t size;
    if (basicsize > PY_SSIZE_T_MAX - sizeof(PyGC_Head))
        return PyErr_NoMemory();
    size = sizeof(PyGC_Head) + basicsize;
    if (use_calloc)
        g = (PyGC_Head *)PyObject_Calloc(1, size);
    else
        g = (PyGC_Head *)PyObject_Malloc(size);
    if (g == NULL)
        return PyErr_NoMemory();
    assert(((uintptr_t)g & 3) == 0);  // g must be aligned 4bytes boundary
    g->_gc_next = 0;
    g->_gc_prev = 0;
    state->generations[0].count++; /* number of allocated GC objects */

    // Determine if collection should occur and which generation to collect.
    if (state->enabled && !state->collecting && !PyErr_Occurred()) {
        // Make a prediction.
        int generation = q_predict(state);

        // Perform collection if necessary.
        if (generation != -1) {
            // Track time of collection.
            // TODO(bobbyluig): Improve timing performance.
            // TODO(bobbyluig): Try different reward shaping.
            _PyTime_t t1 = _PyTime_GetMonotonicClock();
            _PyRuntime.gc.collecting = 1;
            collect_with_callback(state, generation);
            _PyRuntime.gc.collecting = 0;
            _PyTime_t t2 = _PyTime_GetMonotonicClock();

            // If there is some entry in the replay history, apply reward shaping. If the replay table switched from
            // locked to unlocked here, this is okay because the reward shaping information is effectively discarded
            // for this entry only.
            if (q_state.replay_index > 0) {
                // Divide by a power of two, which takes less cycles. This is applied even under full compliance mode
                // in GCC. See https://gcc.gnu.org/wiki/FloatingPointMath.
                q_last_replay()->reward -= _PyTime_AsSecondsDouble(t2 - t1) / 128;
            }
        }
    }

    op = FROM_GC(g);
    return op;
}

PyObject *
_PyObject_GC_Malloc(size_t basicsize)
{
    return _PyObject_GC_Alloc(0, basicsize);
}

PyObject *
_PyObject_GC_Calloc(size_t basicsize)
{
    return _PyObject_GC_Alloc(1, basicsize);
}

PyObject *
_PyObject_GC_New(PyTypeObject *tp)
{
    PyObject *op = _PyObject_GC_Malloc(_PyObject_SIZE(tp));
    if (op != NULL)
        op = PyObject_INIT(op, tp);
    return op;
}

PyVarObject *
_PyObject_GC_NewVar(PyTypeObject *tp, Py_ssize_t nitems)
{
    size_t size;
    PyVarObject *op;

    if (nitems < 0) {
        PyErr_BadInternalCall();
        return NULL;
    }
    size = _PyObject_VAR_SIZE(tp, nitems);
    op = (PyVarObject *) _PyObject_GC_Malloc(size);
    if (op != NULL)
        op = PyObject_INIT_VAR(op, tp, nitems);
    return op;
}

PyVarObject *
_PyObject_GC_Resize(PyVarObject *op, Py_ssize_t nitems)
{
    const size_t basicsize = _PyObject_VAR_SIZE(Py_TYPE(op), nitems);
    _PyObject_ASSERT((PyObject *)op, !_PyObject_GC_IS_TRACKED(op));
    if (basicsize > PY_SSIZE_T_MAX - sizeof(PyGC_Head)) {
        return (PyVarObject *)PyErr_NoMemory();
    }

    PyGC_Head *g = AS_GC(op);
    g = (PyGC_Head *)PyObject_REALLOC(g,  sizeof(PyGC_Head) + basicsize);
    if (g == NULL)
        return (PyVarObject *)PyErr_NoMemory();
    op = (PyVarObject *) FROM_GC(g);
    Py_SIZE(op) = nitems;
    return op;
}

void
PyObject_GC_Del(void *op)
{
    PyGC_Head *g = AS_GC(op);
    if (_PyObject_GC_IS_TRACKED(op)) {
        gc_list_remove(g);
    }
    struct _gc_runtime_state *state = &_PyRuntime.gc;
    if (state->generations[0].count > 0) {
        state->generations[0].count--;
    }
    PyObject_FREE(g);
}
