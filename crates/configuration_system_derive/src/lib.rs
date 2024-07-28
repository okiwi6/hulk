use proc_macro2::TokenStream;
use proc_macro_error::{abort, proc_macro_error};
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Result};

#[proc_macro_derive(Merge)]
#[proc_macro_error]
pub fn merge(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    derive_merge(input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

fn derive_merge(input: DeriveInput) -> Result<TokenStream> {
    let fields = match input.data {
        Data::Struct(data) => data.fields,
        Data::Enum(data) => abort!(data.enum_token, "`Merge` can only be derived for `struct`",),
        Data::Union(data) => abort!(data.union_token, "`Merge` can only be derived for `struct`",),
    };

    let name = input.ident;

    let items = fields.iter().map(|field| {
        let name = &field.ident;

        quote! {
            self.#name.merge(other.#name);
        }
    });

    let where_clauses = fields.iter().map(|fields| {
        let ty = &fields.ty;
        quote! {
            #ty: configuration_system::Merge
        }
    });

    Ok(quote! {
        impl configuration_system::Merge for #name where #(#where_clauses),* {
            fn merge(&mut self, other: Self) {
                #(
                    #items
                )*
            }
        }
    })
}
