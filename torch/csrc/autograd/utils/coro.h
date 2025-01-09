#include <coroutine>

namespace torch {
namespace autograd {
namespace utils {

namespace coro_task_private {

using generic_coroutine_handle = std::coroutine_handle<void>;

struct final_awaiter {
  final_awaiter(generic_coroutine_handle coroutine)
      : caller_coroutine_(coroutine) {}
  ~final_awaiter() {}

  final_awaiter(const final_awaiter &) = delete;
  final_awaiter(final_awaiter &&other) : caller_coroutine_(nullptr) {
    std::swap(caller_coroutine_, other.caller_coroutine_);
  }

  constexpr bool await_ready() const noexcept { return false; }
  auto await_suspend(generic_coroutine_handle) const noexcept {
    return caller_coroutine_;
  }
  void await_resume() const noexcept {}

private:
  generic_coroutine_handle caller_coroutine_;
};


struct promise_base {
  promise_base()
      : handle_(nullptr), parent_(nullptr), leaf_(nullptr), root_(nullptr) {}
  ~promise_base() {}

  promise_base(const promise_base &) = delete;
  promise_base(promise_base &&) = delete;

  auto initial_suspend() { return std::suspend_always{}; }
  auto final_suspend() noexcept {
    if (!parent_) {
        return coro_task_private::final_awaiter(
            std::noop_coroutine());
    }
    
    assert(root_);
    root_->leaf_ = parent_;
    return coro_task_private::final_awaiter(
            parent_->get_coro_handle());
  }
  void unhandled_exception() { std::terminate(); }

  inline void set_parent(promise_base * caller_promise) {
      parent_ = caller_promise;

      assert(parent_);
      root_ = parent_->root_;

      assert(root_);
      root_->leaf_ = this;
  }

  inline promise_base *get_leaf() const {
      assert(leaf_);
      return leaf_;
  }

  inline void set_as_root() {
      leaf_ = this;
      root_ = this;
  }

  inline generic_coroutine_handle get_coro_handle() const {
      return handle_;
  }

protected:
  generic_coroutine_handle handle_;
  promise_base * parent_;
  promise_base * leaf_;
  promise_base * root_;
};

} // namespace coro_task_private

template <typename T> class [[nodiscard]] task {
public:
  struct promise_type;
  struct awaiter;
  using coroutine_handle = std::coroutine_handle<promise_type>;


  task() : coroutine_(nullptr) {}
  task(coroutine_handle coro) : coroutine_(coro) {}
  ~task() {
    if (coroutine_) {
      destroy();
    }
  }

  task(task &&other) : coroutine_(nullptr) {
    std::swap(coroutine_, other.coroutine_);
  }
  task(const task &) = delete;

  task & operator=(task &&other) {
    if (coroutine_) {
      destroy();
    }

    coroutine_ = other.coroutine_;
    other.coroutine_ = nullptr;
    return *this;
  }
  task & operator=(const task &other) = delete;

  bool valid() const {
      return coroutine_ != nullptr;
  }

  bool done() const {
    assert(coroutine_);
    return coroutine_.done();
  }

  void start() {
    assert(coroutine_);
    assert(!coroutine_.done());
    coroutine_.promise().set_as_root();
    coroutine_.resume();
  }

  void resume() {
    assert(coroutine_);
    assert(!coroutine_.done());
    assert(coroutine_.promise().get_leaf());
    assert(coroutine_.promise().get_leaf()->get_coro_handle());
    assert(!coroutine_.promise().get_leaf()->get_coro_handle().done());

    coroutine_.promise().get_leaf()->get_coro_handle().resume();
  }

  void destroy() {
    assert(done());
    coroutine_.destroy();
    coroutine_ = nullptr;
  }

  awaiter operator co_await() const {
    return awaiter(coroutine_);
  }

  template<typename U = T>
  typename std::enable_if<not std::is_same<U, void>::value, U>::type 
  get_return_value() const {
    return coroutine_.promise().transfer_return_value();
  }

private:
  coroutine_handle coroutine_;
};

template <> struct task<void>::promise_type : coro_task_private::promise_base {
  using coroutine_handle =
      std::coroutine_handle<typename task<void>::promise_type>;

  promise_type() {}
  ~promise_type() {}

  auto get_return_object() {
    auto coroutine_handle = coroutine_handle::from_promise(*this);
    handle_ = coroutine_handle;
    return task{coroutine_handle};
  }

  void return_void() {};
};

template <typename T>
struct task<T>::promise_type : coro_task_private::promise_base {
  using coroutine_handle =
      std::coroutine_handle<typename task<T>::promise_type>;

  friend struct task<T>::awaiter;

  promise_type() {}
  ~promise_type() {
    reinterpret_cast<T*>(&ret_val_buf_)->~T();
  }

  auto get_return_object() {
    auto coroutine_handle = coroutine_handle::from_promise(*this);
    handle_ = coroutine_handle;
    return task{coroutine_handle};
  }

  // XXX: explore if there is anyway to get ride of
  // the copy constructing.
  void return_value(T value) {
    new (&ret_val_buf_) T(std::move(value));
  }
  T && transfer_return_value() {
      return std::move(*reinterpret_cast<T*>(&ret_val_buf_));
  }

private:
  struct alignas(alignof(T)) T_Buf {
    uint8_t buf[sizeof(T)];
  };
  T_Buf ret_val_buf_;
};

template <typename T> struct task<T>::awaiter {
  using coroutine_handle =
      std::coroutine_handle<typename task<T>::promise_type>;

  awaiter(coroutine_handle task_coroutine)
      : suspended_task_coroutine_(task_coroutine) {}
  ~awaiter() {}

  awaiter(const awaiter &) = delete;
  awaiter(awaiter && other) : suspended_task_coroutine_(nullptr){
      std::swap(suspended_task_coroutine_, other.suspended_task_coroutine_);
  }

  template <typename awaiting_promise_t>
  auto await_suspend(std::coroutine_handle<awaiting_promise_t>
                         awaiting_coroutine) noexcept {
    suspended_task_coroutine_.promise().set_parent(&(awaiting_coroutine.promise()));
    return suspended_task_coroutine_;
  }
  constexpr bool await_ready() const noexcept {
    return suspended_task_coroutine_.done();
  }

  template <typename U>
  using non_void_T =
      typename std::enable_if<not std::is_same<U, void>::value, U>::type;
  template <typename U>
  using void_T =
      typename std::enable_if<std::is_same<U, void>::value, void>::type;

  template <typename U = T> non_void_T<U> await_resume() noexcept {
    assert(suspended_task_coroutine_.done());
    return suspended_task_coroutine_.promise().transfer_return_value();
  }

  template <typename U = T> void_T<U> await_resume() noexcept {
    assert(suspended_task_coroutine_.done());
  }

private:
  coroutine_handle suspended_task_coroutine_;
};

} // namespace utils
} // namespace autograd
} // namespace torch