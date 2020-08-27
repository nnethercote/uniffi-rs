// This file was autogenerated by some hot garbage in the `uniffi` crate.
// Trust me, you don't want to mess with it!

{% import "macros.cpp" as cpp %}

#ifndef mozilla_dom_{{ obj.name()|class_name_webidl }}
#define mozilla_dom_{{ obj.name()|class_name_webidl }}

#include "jsapi.h"
#include "nsCOMPtr.h"
#include "nsIGlobalObject.h"
#include "nsWrapperCache.h"

#include "mozilla/RefPtr.h"

#include "mozilla/dom/{{ namespace|class_name_webidl }}Binding.h"

namespace mozilla {
namespace dom {

class {{ obj.name()|class_name_cpp }} final : public nsISupports, public nsWrapperCache {
 public:
  NS_DECL_CYCLE_COLLECTING_ISUPPORTS
  NS_DECL_CYCLE_COLLECTION_SCRIPT_HOLDER_CLASS({{ obj.name()|class_name_cpp }})

  {{ obj.name()|class_name_cpp }}(nsIGlobalObject* aGlobal, int64_t aHandle);

  JSObject* WrapObject(JSContext* aCx,
                       JS::Handle<JSObject*> aGivenProto) override;

  nsIGlobalObject* GetParentObject() const { return mGlobal; }

  {% for cons in obj.constructors() %}
  static already_AddRefed<{{ obj.name()|class_name_cpp }}> Constructor(
    {% call cpp::decl_constructor_args(cons) %}
  );
  {%- endfor %}

  {% for meth in obj.methods() %}
  {% call cpp::decl_return_type(meth) %} {{ meth.name()|fn_name_cpp }}(
    {% call cpp::decl_method_args(meth) %}
  );
  {% endfor %}

 private:
  ~{{ obj.name()|class_name_cpp }}();

  nsCOMPtr<nsIGlobalObject> mGlobal;
  int64_t mHandle;
};

}  // namespace dom
}  // namespace mozilla

#endif  // mozilla_dom_{{ obj.name()|class_name_webidl }}
