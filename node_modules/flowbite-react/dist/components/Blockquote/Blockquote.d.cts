import { type ComponentProps } from "react";
import type { ThemingProps } from "../../types";
export interface BlockquoteTheme {
    root: BlockquoteRootTheme;
}
export interface BlockquoteRootTheme {
    base: string;
}
export interface BlockquoteProps extends ComponentProps<"blockquote">, ThemingProps<BlockquoteTheme> {
}
export declare const Blockquote: import("react").ForwardRefExoticComponent<Omit<BlockquoteProps, "ref"> & import("react").RefAttributes<HTMLQuoteElement>>;
