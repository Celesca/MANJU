import type { ComponentProps, ReactNode } from "react";
import type { FlowbiteBoolean, ThemingProps } from "../../types";
export interface CarouselTheme {
    root: CarouselRootTheme;
    indicators: CarouselIndicatorsTheme;
    item: CarouselItemTheme;
    control: CarouselControlTheme;
    scrollContainer: CarouselScrollContainer;
}
export interface CarouselRootTheme {
    base: string;
    leftControl: string;
    rightControl: string;
}
export interface CarouselIndicatorsTheme {
    active: FlowbiteBoolean;
    base: string;
    wrapper: string;
}
export interface CarouselItemTheme {
    base: string;
    wrapper: FlowbiteBoolean;
}
export interface CarouselControlTheme {
    base: string;
    icon: string;
}
export interface CarouselScrollContainer {
    base: string;
    snap: string;
}
export interface CarouselProps extends ComponentProps<"div">, ThemingProps<CarouselTheme> {
    indicators?: boolean;
    leftControl?: ReactNode;
    rightControl?: ReactNode;
    draggable?: boolean;
    slide?: boolean;
    slideInterval?: number;
    onSlideChange?: (slide: number) => void;
    pauseOnHover?: boolean;
}
export interface DefaultLeftRightControlProps extends ComponentProps<"div">, ThemingProps<CarouselTheme> {
}
export declare const Carousel: import("react").ForwardRefExoticComponent<Omit<CarouselProps, "ref"> & import("react").RefAttributes<HTMLDivElement>>;
