import type { ComponentProps } from "react";
import type { DynamicStringEnumKeysOf, ThemingProps } from "../../types";
import type { TextInputColors, TextInputSizes } from "../TextInput";
export interface FileInputTheme {
    base: string;
    colors: TextInputColors;
    sizes: TextInputSizes;
}
export interface FileInputProps extends Omit<ComponentProps<"input">, "type" | "ref" | "color">, ThemingProps<FileInputTheme> {
    color?: DynamicStringEnumKeysOf<TextInputColors>;
    sizing?: DynamicStringEnumKeysOf<TextInputSizes>;
}
export declare const FileInput: import("react").ForwardRefExoticComponent<FileInputProps & import("react").RefAttributes<HTMLInputElement>>;
